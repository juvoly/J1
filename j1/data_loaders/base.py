from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Optional
import time

import dask.dataframe as dd
import pandas as pd
from tqdm import tqdm


class BaseDataLoader(ABC):
    """Base class for all data loaders.
    
    A data loader is responsible for loading data from a source and yielding documents
    in a structured format. The documents are then collected into a Dask DataFrame.
    """
    
    def __init__(self, shard_size: int = 1000):
        """Initialize the data loader.
        
        Args:
            shard_size: Number of documents to include in each shard.
        """
        self.shard_size = shard_size
        self._current_shard: list[Dict[str, Any]] = []
        self._total_documents = 0
        self._start_time = None
    
    @abstractmethod
    async def load_documents(self) -> AsyncIterator[Dict[str, Any]]:
        """Load documents from the source.
        
        Yields:
            Dictionary containing document data.
        """
        pass
    
    async def _yield_shard(self) -> Optional[pd.DataFrame]:
        """Convert the current shard to a DataFrame and clear it.
        
        Returns:
            DataFrame containing the shard's documents, or None if the shard is empty.
        """
        if not self._current_shard:
            return None
        
        df = pd.DataFrame(self._current_shard)
        self._current_shard = []
        return df
    
    async def to_dask_dataframe(self) -> dd.DataFrame:
        """Convert the loaded documents to a Dask DataFrame.
        
        Returns:
            Dask DataFrame containing all documents.
        """
        dfs = []
        self._start_time = time.time()
        self._total_documents = 0
        
        async for doc in self.load_documents():
            self._total_documents += 1
            self._current_shard.append(doc)
            
            elapsed = time.time() - self._start_time
            docs_per_second = self._total_documents / elapsed if elapsed > 0 else 0
            
            print(f"\rProcessed {self._total_documents} documents ({docs_per_second:.1f} docs/s)", end="")
            
            if len(self._current_shard) >= self.shard_size:
                df = await self._yield_shard()
                if df is not None:
                    dfs.append(df)
        
        if self._current_shard:
            df = await self._yield_shard()
            if df is not None:
                dfs.append(df)
                
                total_elapsed = time.time() - self._start_time
                avg_docs_per_second = self._total_documents / total_elapsed if total_elapsed > 0 else 0
                
                print(f"\rProcessed {self._total_documents} documents (avg: {avg_docs_per_second:.1f} docs/s)")
        
        if not dfs:
            return dd.from_pandas(pd.DataFrame(), npartitions=1)
        
        dask_dfs = [dd.from_pandas(df, npartitions=1) for df in dfs]
        return dd.concat(dask_dfs) 