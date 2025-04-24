from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional
import dask.dataframe as dd
import pandas as pd


class BaseDataLoader(ABC):
    """Base class for data loaders."""
    
    def __init__(self):
        """Initialize the data loader."""
        pass
    
    @abstractmethod
    async def load_documents(self) -> AsyncIterator[Dict[str, Any]]:
        """Load documents from the data source.
        
        Yields:
            Dictionary containing document data.
        """
        pass
    
    async def to_dask_dataframe(self) -> dd.DataFrame:
        """Convert the loaded documents to a Dask DataFrame.
        
        Returns:
            Dask DataFrame containing all documents.
        """
        docs = []
        async for doc in self.load_documents():
            if isinstance(doc, dd.DataFrame):
                docs.append(doc)
            elif isinstance(doc, pd.DataFrame):
                docs.append(dd.from_pandas(doc, npartitions=1))
        
        if not docs:
            return dd.from_pandas(pd.DataFrame(), npartitions=1)
        
        return dd.concat(docs)