import asyncio
import gzip
import json
import tarfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aiohttp import ClientResponse, ClientSession
from bs4 import BeautifulSoup
import tempfile

from j1.data_loaders import PubMedDataLoader

# Constants
BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/"

# Mock HTML content for the FTP index page
MOCK_HTML = """
<html>
<body>
    <a href="small.tar.gz">small.tar.gz</a> 2024-01-01 12:00 1M
    <a href="medium.tar.gz">medium.tar.gz</a> 2024-01-01 12:00 2M
    <a href="large.tar.gz">large.tar.gz</a> 2024-01-01 12:00 3M
</body>
</html>
"""

# Mock XML content for articles
MOCK_ARTICLE_XML = """
<article>
    <front>
        <article-meta>
            <title-group>
                <article-title>Test Article Title</article-title>
            </title-group>
            <abstract>
                <p>This is a test abstract.</p>
            </abstract>
        </article-meta>
    </front>
    <body>
        <sec>
            <title>Introduction</title>
            <p>This is the introduction.</p>
        </sec>
        <sec>
            <title>Methods</title>
            <p>These are the methods.</p>
        </sec>
    </body>
</article>
"""


@pytest.fixture
def mock_tar_file(tmp_path):
    """Create a mock tar.gz file with XML articles."""
    tar_path = tmp_path / "test.tar.gz"
    
    # Create a tar file with mock articles
    with tarfile.open(tar_path, "w:gz") as tar:
        # Add mock XML files
        for i in range(3):
            xml_content = MOCK_ARTICLE_XML.replace("Test Article Title", f"Test Article {i}")
            xml_path = tmp_path / f"article_{i}.xml"
            xml_path.write_text(xml_content)
            tar.add(xml_path, arcname=f"article_{i}.xml")
    
    return tar_path


class MockResponse:
    """Mock response class for testing."""
    
    def __init__(self, text, paragraphs=None, content=None):
        self._text = text
        self._paragraphs = paragraphs or []
        self._content = content
        self.status = 200
        
    async def text(self):
        return self._text
        
    async def json(self):
        return {"esearchresult": {"idlist": ["12345678", "87654321"]}}
        
    @property
    def content(self):
        return self
        
    async def iter_chunked(self, chunk_size):
        """Simulate streaming response."""
        if self._content:
            yield self._content
        else:
            yield self._text.encode()
        
    def __iter__(self):
        """Return an iterator over the paragraphs."""
        return iter(self._paragraphs)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockClientSession:
    """Mock aiohttp ClientSession that supports async context manager."""
    def __init__(self, mock_tar_file):
        self.mock_tar_file = mock_tar_file

    async def get(self, url, **kwargs):
        """Mock get method that returns a mock response."""
        if url == BASE_URL:
            # Return the FTP index page
            return MockResponse(MOCK_HTML)
            
        # Handle tar file download
        if url.endswith('.tar.gz'):
            if self.mock_tar_file:
                with open(self.mock_tar_file, 'rb') as f:
                    content = f.read()
                return MockResponse("", content=content)
            return MockResponse("")
            
        # Handle article requests
        try:
            article_num = int(url.split("/")[-1])
        except ValueError:
            # If we can't parse the article number, return empty content
            return MockResponse("")
        
        # Create a mock article with paragraphs
        paragraphs = [
            f"Introduction for article {article_num}",
            f"Methods for article {article_num}",
            f"Results for article {article_num}",
            f"Discussion for article {article_num}"
        ]
        
        # Create a mock article XML
        article_xml = f"""
        <PubmedArticle>
            <MedlineCitation>
                <Article>
                    <ArticleTitle>Test Article {article_num}</ArticleTitle>
                    <Abstract>
                        <AbstractText>This is a test abstract for article {article_num}</AbstractText>
                    </Abstract>
                </Article>
            </MedlineCitation>
            <PubmedData>
                <ArticleIdList>
                    <ArticleId IdType="pubmed">{article_num}</ArticleId>
                </ArticleIdList>
            </PubmedData>
        </PubmedArticle>
        """
        
        return MockResponse(article_xml, paragraphs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_session(mock_tar_file):
    """Create a mock aiohttp session."""
    return MockClientSession(mock_tar_file)


@pytest.mark.asyncio
async def test_pubmed_loader(mock_session, tmp_path):
    loader = PubMedDataLoader(
        shard_size=2,
        concurrency=1,
        article_limit=3
    )
    
    with patch("aiohttp.ClientSession", return_value=mock_session):
        df = await loader.to_dask_dataframe()
        
        expected_partitions = 2
        assert df.npartitions == expected_partitions, f"Expected {expected_partitions} partitions, got {df.npartitions}"
        
        pdf = df.compute()
        
        assert len(pdf) == 3
        assert "title" in pdf.columns
        assert "abstract" in pdf.columns
        assert "body" in pdf.columns
        assert "filename" in pdf.columns
        
        first_article = pdf.iloc[0]
        assert "Test Article 0" in first_article["title"]
        assert "test abstract" in first_article["abstract"].lower()
        assert "introduction" in first_article["body"].lower()
        assert "methods" in first_article["body"].lower()


@pytest.mark.asyncio
async def test_pubmed_loader_error_handling(mock_session):
    """Test error handling in the PubMed data loader."""
    # Create a session that simulates errors
    error_session = MockClientSession(None)
    
    # Create the loader
    loader = PubMedDataLoader(
        shard_size=2,
        concurrency=1,
        article_limit=3
    )
    
    # Mock the aiohttp.ClientSession
    with patch("aiohttp.ClientSession", return_value=error_session):
        # Get the Dask DataFrame
        df = await loader.to_dask_dataframe()
        
        # Convert to pandas for easier testing
        pdf = df.compute()
        
        # Should still return an empty DataFrame without raising an exception
        assert len(pdf) == 0


async def test_fetch_article():
    """Test fetching a single article."""
    loader = PubMedDataLoader()
    article_id = "12345678"
    
    # Create a mock tar file
    with tempfile.NamedTemporaryFile(suffix='.tar.gz') as tmp:
        # Create a tar file with mock articles
        with tarfile.open(tmp.name, "w:gz") as tar:
            # Add mock XML files
            for i in range(3):
                xml_content = MOCK_ARTICLE_XML.replace("Test Article Title", f"Test Article {i}")
                xml_path = Path(tmp.name).parent / f"article_{i}.xml"
                xml_path.write_text(xml_content)
                tar.add(xml_path, arcname=f"article_{i}.xml")
                xml_path.unlink()
        
        # Create a mock session that handles both FTP and article requests
        mock_session = MockClientSession(tmp.name)
        
        # Mock the aiohttp.ClientSession
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Get the Dask DataFrame
            df = await loader.to_dask_dataframe()
            
            # Convert to pandas for testing
            pdf = df.compute()
            
            # Verify the article content
            assert len(pdf) > 0
            first_article = pdf.iloc[0]
            assert "Test Article" in first_article["title"]
            assert "test abstract" in first_article["abstract"].lower()
            assert "introduction" in first_article["body"].lower()
            assert "methods" in first_article["body"].lower() 