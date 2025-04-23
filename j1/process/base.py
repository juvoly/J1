from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import dask.dataframe as dd


class Processor(ABC):
    """Abstract base class for processing dask dataframes."""
    
    @abstractmethod
    def process(self, df: dd.DataFrame) -> dd.DataFrame:
        """Process the input dataframe.
        
        Args:
            df: Input dask dataframe
            
        Returns:
            Processed dask dataframe
        """
        pass


class TextProcessor(Processor):
    """Base class for text processing operations on dask dataframes."""
    
    def __init__(self, text_key: str):
        """Initialize the text processor.
        
        Args:
            text_key: Name of the column containing the text to process
        """
        self.text_key = text_key
    
    def process(self, df: dd.DataFrame) -> dd.DataFrame:
        """Process the input dataframe.
        
        Args:
            df: Input dask dataframe
            
        Returns:
            Processed dask dataframe
        """
        if self.text_key not in df.columns:
            raise ValueError(f"Column '{self.text_key}' not found in dataframe")
        return self._process_text(df)
    
    @abstractmethod
    def _process_text(self, df: dd.DataFrame) -> dd.DataFrame:
        """Process the text column of the dataframe.
        
        Args:
            df: Input dask dataframe
            
        Returns:
            Processed dask dataframe
        """
        pass 