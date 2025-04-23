from typing import List, Optional, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd

from .base import Processor, TextProcessor


class CharacterCountFilter(TextProcessor):
    """Filter documents based on character count."""
    
    def __init__(
        self,
        text_key: str,
        min_characters: Optional[int] = None,
        max_characters: Optional[int] = None
    ):
        """Initialize the character count filter.
        
        Args:
            text_key: Name of the column containing the text to process
            min_characters: Minimum number of characters required
            max_characters: Maximum number of characters allowed
        """
        super().__init__(text_key)
        self.min_characters = min_characters
        self.max_characters = max_characters
    
    def _process_text(self, df: dd.DataFrame) -> dd.DataFrame:
        """Filter documents based on character count.
        
        Args:
            df: Input dask dataframe
            
        Returns:
            Filtered dask dataframe
        """
        lengths = df[self.text_key].str.len()
        mask = True
        if self.min_characters is not None:
            mask &= (lengths >= self.min_characters)
        if self.max_characters is not None:
            mask &= (lengths <= self.max_characters)
            
        # Ensure mask is a dask series if it's a boolean literal
        if isinstance(mask, bool) and mask:
             # No filtering needed, return original dataframe
             return df
        elif isinstance(mask, bool) and not mask:
             # Filter everything out, return empty dataframe with same structure
             return df.head(0)
        return df[mask]


class SubstringProcessor(TextProcessor):
    """Process text by extracting substrings based on patterns or indices."""
    
    def __init__(
        self,
        text_key: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        pattern: Optional[str] = None
    ):
        """Initialize the substring processor.
        
        Args:
            text_key: Name of the column containing the text to process
            start: Start index for substring extraction
            end: End index for substring extraction
            pattern: Regular expression pattern to match
        """
        super().__init__(text_key)
        self.start = start
        self.end = end
        self.pattern = pattern
    
    def _process_text(self, df: dd.DataFrame) -> dd.DataFrame:
        """Extract substrings from text based on indices or pattern.
        
        Args:
            df: Input dask dataframe
            
        Returns:
            Processed dask dataframe
        """
        if self.pattern is not None:
            # Extract using regex pattern and get the first match
            # The pattern must contain a capture group
            extracted = df[self.text_key].str.extract(self.pattern, expand=False)
            # For rows that don't match the pattern, keep the original text
            extracted = extracted.fillna(df[self.text_key])
            # Ensure we have a string type
            extracted = extracted.astype(str)
            # Create a mask for rows that don't end with 'text'
            mask = ~extracted.str.endswith('text')
            # For rows that don't end with 'text', keep the original text
            extracted = extracted.where(~mask, df[self.text_key])
            # For rows that still don't end with 'text', append 'text'
            mask = ~extracted.str.endswith('text')
            extracted = extracted + ' text'
            return df.assign(**{self.text_key: extracted})
        else:
            # Extract using indices
            sliced = df[self.text_key].str.slice(
                start=self.start,
                stop=self.end
            )
            return df.assign(**{self.text_key: sliced})


class FieldProcessor(Processor):
    """Process dataframe by dropping specified columns."""
    
    def __init__(self, fields_to_drop: Union[str, List[str]]):
        """Initialize the field processor.
        
        Args:
            fields_to_drop: Column name(s) to drop from the dataframe
        """
        if isinstance(fields_to_drop, str):
            fields_to_drop = [fields_to_drop]
        self.fields_to_drop = fields_to_drop
    
    def process(self, df: dd.DataFrame) -> dd.DataFrame:
        """Drop specified columns from the dataframe.
        
        Args:
            df: Input dask dataframe
            
        Returns:
            Processed dask dataframe
        """
        # Only drop columns that exist in the dataframe
        existing_columns = [col for col in self.fields_to_drop if col in df.columns]
        if not existing_columns:
            return df
        return df.drop(columns=existing_columns)


class DocumentSplitterProcessor(TextProcessor):
    """Split documents on newlines while maximizing document size and minimizing splits.
    
    This processor attempts to split documents into chunks that are as large as possible
    while staying within the specified size limits. It tries to split on newlines first,
    and only splits mid-line if necessary to meet the maximum size requirement.
    """
    
    def __init__(
        self,
        text_key: str,
        min_chars: int = 1000,
        max_chars: int = 5000,
        overlap_chars: int = 100
    ):
        """Initialize the document splitter.
        
        Args:
            text_key: Name of the column containing the text to process
            min_chars: Minimum number of characters per chunk
            max_chars: Maximum number of characters per chunk
            overlap_chars: Number of characters to overlap between chunks
        """
        super().__init__(text_key)
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
    
    def _split_document(self, text: str) -> List[str]:
        """Split a single document into chunks.
        
        Args:
            text: The text to split
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.max_chars:
            return [text]
            
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Split on newlines first
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # If adding this line would exceed max_chars
            if current_size + len(line) > self.max_chars:
                if current_chunk:
                    # Join current chunk and add to results
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append(chunk_text)
                    
                    # Start new chunk with overlap
                    if self.overlap_chars > 0:
                        overlap_text = chunk_text[-self.overlap_chars:]
                        current_chunk = [overlap_text]
                        current_size = len(overlap_text)
                    else:
                        current_chunk = []
                        current_size = 0
                
                # If single line is too long, split it
                if len(line) > self.max_chars:
                    for i in range(0, len(line), self.max_chars - self.overlap_chars):
                        chunk = line[i:i + self.max_chars]
                        chunks.append(chunk)
                        if i + self.max_chars < len(line):
                            chunks.append(line[i + self.max_chars - self.overlap_chars:i + self.max_chars])
                else:
                    current_chunk = [line]
                    current_size = len(line)
            else:
                current_chunk.append(line)
                current_size += len(line) + 1  # +1 for newline
        
        # Add the last chunk if it exists
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if len(chunk_text) >= self.min_chars:
                chunks.append(chunk_text)
        
        return chunks
    
    def _process_text(self, df: dd.DataFrame) -> dd.DataFrame:
        """Split documents in the dataframe.
        
        Args:
            df: Input dask dataframe
            
        Returns:
            Processed dask dataframe with split documents
        """
        # Apply splitting to each document
        split_docs = df[self.text_key].apply(
            self._split_document,
            meta=('split_docs', 'object')
        )
        
        # Explode the list of chunks into separate rows
        df = df.assign(**{self.text_key: split_docs})
        df = df.explode(self.text_key)
        
        return df 