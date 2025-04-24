import asyncio
import logging
import tarfile
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional
from urllib.parse import urljoin
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError

import aiofiles
import aiohttp
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
import pandas as pd
import dask.dataframe as dd

from .base import BaseDataLoader

logger = logging.getLogger(__name__)

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/"
DEFAULT_CONCURRENCY = 5


def get_element_text(element: ET.Element | None) -> str:
    """Recursively extracts all text from an XML element and its children."""
    if element is None:
        return ""
    
    element_copy = ET.Element(element.tag, element.attrib)
    element_copy.text = element.text
    element_copy.tail = element.tail
    
    # Remove references, tables, figures, and labels
    for tag in ['.//xref', './/table', './/table-wrap', './/fig', './/label']:
        for elem in element_copy.findall(tag):
            parent = elem.getparent()
            if parent is not None:
                parent.remove(elem)
    
    text = "".join(element_copy.itertext()).strip()
    return text.replace('[', '').replace(']', '')


def extract_article_content(xml_content: str) -> Dict[str, str]:
    """Extract title, abstract, and body from XML content."""
    try:
        root = ET.fromstring(xml_content)
        
        # Extract title
        title_elem = root.find('.//article-title')
        title_text = get_element_text(title_elem)
        
        # Extract abstract
        abstract_elem = root.find('.//abstract')
        abstract_text = ""
        if abstract_elem is not None:
            abstract_paragraphs = abstract_elem.findall('.//p')
            abstract_text = "\n".join(get_element_text(p) for p in abstract_paragraphs if p is not None)
        
        # Extract body paragraphs
        body = root.find('.//body')
        paragraphs = []
        if body is not None:
            for sec in body.findall('.//sec'):
                title_elem = sec.find('.//title')
                sec_title_text = get_element_text(title_elem).lower()
                if title_elem is not None and sec_title_text is not None and ('author' in sec_title_text or 'contribution' in sec_title_text):
                    break
                paragraphs.append(sec_title_text)
                for p in sec.findall('.//p'):
                    p_text = get_element_text(p)
                    if p_text:
                        paragraphs.append(p_text)
            
            if not paragraphs:
                for p in body.findall('./p'):
                    p_text = get_element_text(p)
                    if p_text:
                        paragraphs.append(p_text)

        return {
            'title': title_text,
            'abstract': abstract_text,
            'body': "\n\n".join(paragraphs) if paragraphs else ""
        }
    except ParseError as e:
        logger.error(f"Error parsing XML: {e}")
        return {'title': '', 'abstract': '', 'body': ''}
    except Exception as e:
        logger.exception(f"Unexpected error processing XML: {e}")
        return {'title': '', 'abstract': '', 'body': ''}


class PubMedDataLoader(BaseDataLoader):
    """Data loader for PubMed Open Access articles."""
    
    def __init__(
        self,
        shard_size: int = 640,
        concurrency: int = DEFAULT_CONCURRENCY,
        article_limit: Optional[int] = None
    ):
        """Initialize the PubMed data loader.
        
        Args:
            shard_size: Number of articles per shard.
            concurrency: Number of concurrent downloads.
            article_limit: Maximum number of articles to process.
        """
        super().__init__()
        self.shard_size = shard_size
        self.concurrency = concurrency
        self.article_limit = article_limit
        self._processed_count = 0
    
    async def _get_ftp_file_list(self, session: aiohttp.ClientSession) -> list[tuple[str, str, int]]:
        """Fetch list of .tar.gz files and their sizes."""
        files = []
        try:
            response = await session.get(BASE_URL)
            async with response:
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                for a_tag in soup.find_all('a'):
                    href = a_tag.get('href')
                    if href and href.endswith('.tar.gz'):
                        filename = href
                        full_url = urljoin(BASE_URL, href)
                        size_str = "0"
                        
                        next_sibling_text = a_tag.next_sibling
                        if next_sibling_text and isinstance(next_sibling_text, str):
                            parts = next_sibling_text.strip().split()
                            if len(parts) >= 2:
                                raw_size = parts[-1].upper()
                                multiplier = 1
                                if raw_size.endswith('K'):
                                    multiplier = 1024
                                    raw_size = raw_size[:-1]
                                elif raw_size.endswith('M'):
                                    multiplier = 1024**2
                                    raw_size = raw_size[:-1]
                                elif raw_size.endswith('G'):
                                    multiplier = 1024**3
                                    raw_size = raw_size[:-1]
                                
                                try:
                                    size_bytes = int(float(raw_size) * multiplier)
                                    size_str = str(size_bytes)
                                except ValueError:
                                    logger.warning(f"Could not parse size for {filename}")
                        
                        files.append((full_url, filename, int(size_str)))
        except Exception as e:
            logger.error(f"Error fetching file list: {e}")
        
        return files
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(aiohttp.ClientError),
        reraise=True
    )
    async def _download_file(self, session: aiohttp.ClientSession, url: str, destination: Path) -> None:
        """Download a single file with retries."""
        response = await session.get(url, timeout=aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=600))
        async with response:
            async with aiofiles.open(destination, mode='wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    await f.write(chunk)
    
    async def _process_archive(
        self,
        session: aiohttp.ClientSession,
        url: str,
        filename: str,
        temp_dir: Path,
        semaphore: asyncio.Semaphore
    ) -> AsyncIterator[Dict[str, Any]]:
        """Process a single archive file."""
        async with semaphore:
            temp_tar_path = temp_dir / filename
            try:
                await self._download_file(session, url, temp_tar_path)
                
                with tarfile.open(temp_tar_path, "r:gz") as tar:
                    for member in tar.getmembers():
                        if self.article_limit is not None and self._processed_count >= self.article_limit:
                            return
                        
                        if member.isfile() and member.name.lower().endswith('.xml'):
                            try:
                                fileobj = tar.extractfile(member)
                                if fileobj:
                                    content = fileobj.read()
                                    article_data = extract_article_content(content.decode('utf-8', errors='replace'))
                                    article_data['filename'] = member.name
                                    self._processed_count += 1
                                    yield article_data
                            except Exception as e:
                                logger.error(f"Error processing {member.name}: {e}")
                
            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")
            finally:
                if temp_tar_path.exists():
                    temp_tar_path.unlink()
    
    async def load_documents(self) -> AsyncIterator[Dict[str, Any]]:
        """Load documents from PubMed Open Access."""
        semaphore = asyncio.Semaphore(self.concurrency)
        connector = aiohttp.TCPConnector(limit_per_host=self.concurrency)
        
        current_shard = []
        async with aiohttp.ClientSession(connector=connector) as session:
            file_list = await self._get_ftp_file_list(session)
            if not file_list:
                return
            
            # Sort by size (smallest first)
            file_list.sort(key=lambda item: item[2])
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                for url, filename, _ in file_list:
                    if self.article_limit is not None and self._processed_count >= self.article_limit:
                        break
                    
                    async for doc in self._process_archive(session, url, filename, temp_path, semaphore):
                        current_shard.append(doc)
                        
                        # If we've reached the shard size, yield the shard as a dask DataFrame
                        if len(current_shard) >= self.shard_size:
                            yield dd.from_pandas(pd.DataFrame(current_shard), npartitions=1)
                            current_shard = []
                
                # Yield any remaining documents in the last shard
                if current_shard:
                    yield dd.from_pandas(pd.DataFrame(current_shard), npartitions=1) 