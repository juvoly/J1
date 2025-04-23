from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional
import pandas as pd
import dask.dataframe as dd
from .base import BaseDataLoader


class LocalDataLoader(BaseDataLoader):
    """Data loader for loading data from local files.
    
    Supports various file formats including parquet, JSON, CSV, etc.
    """
    
    def __init__(
        self,
        path: str,
        format: str = "parquet",
        **kwargs
    ):
        """Initialize the local data loader.
        
        Args:
            path: Path to the file or directory containing the data
            format: File format (parquet, json, csv, etc.)
            **kwargs: Additional arguments passed to the base class
        """
        super().__init__(**kwargs)
        self.path = Path(path)
        self.format = format.lower()
        
        if not self.path.exists():
            raise FileNotFoundError(f"Path {path} does not exist")
    
    async def load_documents(self) -> AsyncIterator[Dict[str, Any]]:
        """Load documents from local files.
        
        Yields:
            Dictionary containing document data.
        """
        if self.format == "parquet":
            if self.path.is_dir():
                # Load from parquet directory
                df = dd.read_parquet(str(self.path))
            else:
                # Load from single parquet file
                df = dd.read_parquet(str(self.path))
        elif self.format == "json":
            if self.path.is_dir():
                # Load from directory of JSON files
                df = dd.read_json(str(self.path / "*.json"))
            else:
                # Load from single JSON file
                df = dd.read_json(str(self.path))
        elif self.format == "csv":
            if self.path.is_dir():
                # Load from directory of CSV files
                df = dd.read_csv(str(self.path / "*.csv"))
            else:
                # Load from single CSV file
                df = dd.read_csv(str(self.path))
        else:
            raise ValueError(f"Unsupported format: {self.format}")
        
        # Convert to pandas DataFrame for processing
        df = df.compute()
        
        # Yield each row as a document
        for _, row in df.iterrows():
            yield row.to_dict() 