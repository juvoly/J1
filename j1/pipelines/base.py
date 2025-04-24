import asyncio
import importlib
import time
from typing import Any, Dict, List, Type, Optional

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar

from j1.data_loaders.base import BaseDataLoader
from j1.process.base import Processor
from j1.process.writers import DataWriter


def _import_class(class_path: str) -> Type:
    """Dynamically import a class from a module path."""
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(f"Failed to import class {class_path}: {e}")


class Pipeline:
    """
    Represents a configurable data processing pipeline.

    Loads data using a specified loader, applies a sequence of processors,
    and saves the results.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline from a configuration dictionary.

        Args:
            config: The pipeline configuration. Expected keys:
                - loader: Dict with 'class' (path) and 'params' (dict).
                - processors: List of dicts, each with 'class' (path) and 'params' (dict).
                - output: Dict with:
                    - path: str (required)
                    - format: str (optional, default 'parquet')
                    - writer: Dict with 'class' (path) and 'params' (dict) (optional)
                - dask_workers: Optional int, number of Dask workers.
        """
        self.config = config
        self._validate_config()

        self.loader_class: Type[BaseDataLoader] = _import_class(self.config['loader']['class'])
        self.loader_params: Dict[str, Any] = self.config['loader'].get('params', {})

        self.processor_configs: List[Dict[str, Any]] = self.config.get('processors', [])
        self.processors: List[Processor] = []
        for proc_conf in self.processor_configs:
            klass: Type[Processor] = _import_class(proc_conf['class'])
            params = proc_conf.get('params', {})
            self.processors.append(klass(**params))

        self.output_path: str = self.config['output']['path']
        self.output_format: str = self.config['output'].get('format', 'parquet')
        self.dask_workers: int | None = self.config.get('dask_workers')

        # Initialize writer if specified
        self.writer: Optional[DataWriter] = None
        if 'writer' in self.config['output']:
            writer_class = _import_class(self.config['output']['writer']['class'])
            writer_params = self.config['output']['writer'].get('params', {})
            self.writer = writer_class(**writer_params)

    def _validate_config(self):
        """Basic validation of the configuration structure."""
        if 'loader' not in self.config or 'class' not in self.config['loader']:
            raise ValueError("Pipeline config must include 'loader' with a 'class'.")
        if 'output' not in self.config or 'path' not in self.config['output']:
            raise ValueError("Pipeline config must include 'output' with a 'path'.")
        if not isinstance(self.config.get('processors', []), list):
             raise ValueError("'processors' must be a list.")

    async def _load_data(self) -> dd.DataFrame:
        """Load data using the configured loader."""
        loader_instance = self.loader_class(**self.loader_params)
        print(f"Loading data using {self.config['loader']['class']}...")
        df = await loader_instance.to_dask_dataframe()
        print(f"Loading complete. DataFrame has {df.npartitions} partitions.")
        return df

    def _process_data(self, df: dd.DataFrame) -> dd.DataFrame:
        """Apply processors sequentially to the dataframe."""
        if not self.processors:
            return df

        print(f"Applying {len(self.processors)} processors...")
        processed_df = df
        for i, processor in enumerate(self.processors):
            print(f"  Applying processor {i+1}/{len(self.processors)}: {processor.__class__.__name__}...")
            processed_df = processor.process(processed_df)
        return processed_df

    def _save_data(self, df: dd.DataFrame):
        """Save the processed dataframe."""
        if self.writer:
            print(f"Writing data using {self.writer.__class__.__name__}...")
            self.writer.write(df)
            print("Writing complete.")
        else:
            print(f"Saving results to {self.output_path} (format: {self.output_format})...")
            with ProgressBar():
                if self.output_format == 'parquet':
                    df.to_parquet(self.output_path)
                elif self.output_format == 'csv':
                    # Note: Saving to single CSV might be slow/memory intensive for large data
                    df.to_csv(self.output_path, single_file=True, index=False)
                else:
                    raise ValueError(f"Unsupported output format: {self.output_format}")
            print("Saving complete.")

    async def run(self):
        """Execute the entire pipeline: load, process, save."""
        cluster = None
        client = None
        try:
            if self.dask_workers:
                print(f"Setting up Dask cluster with {self.dask_workers} workers...")
                cluster = LocalCluster(n_workers=self.dask_workers)
                client = Client(cluster)
                print(f"Dask dashboard link: {client.dashboard_link}")
            else:
                print("Running Dask operations sequentially (no explicit cluster).")

            start_time = time.time()

            # Load
            raw_df = await self._load_data()

            # Process
            processed_df = self._process_data(raw_df)

            # Save
            self._save_data(processed_df)

            end_time = time.time()
            print(f"Pipeline finished in {end_time - start_time:.2f} seconds.")

        finally:
            if client:
                client.close()
            if cluster:
                cluster.close()
            print("Dask cluster shut down.") 