import asyncio
import importlib
import json
import yaml
import os
from pathlib import Path
from typing import Any, Dict, Type
import re

import click
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from dotenv import load_dotenv

from .data_loaders.base import BaseDataLoader
from .pipelines.base import Pipeline
from .eval.medqa import MultiModelEvaluator


def load_env_vars():
    """Load environment variables from .env file."""
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
    else:
        click.echo("Warning: .env file not found. Using system environment variables.", err=True)


def substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively substitute environment variables in the configuration."""
    if isinstance(config, dict):
        return {k: substitute_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [substitute_env_vars(item) for item in config]
    elif isinstance(config, str):
        # Match ${VAR_NAME} pattern
        env_var_pattern = r'\${([^}]+)}'
        matches = re.findall(env_var_pattern, config)
        if matches:
            for var_name in matches:
                env_value = os.getenv(var_name)
                if env_value is None:
                    raise ValueError(f"Environment variable {var_name} not found")
                config = config.replace(f"${{{var_name}}}", env_value)
        return config
    return config


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file and substitute environment variables."""
    # First load environment variables
    load_env_vars()
    
    path = Path(config_path)
    if not path.exists():
        raise click.BadParameter(f"Config file {config_path} does not exist")
    
    with open(path) as f:
        if path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise click.BadParameter(f"Unsupported config file format: {path.suffix}")
    
    # Substitute environment variables
    return substitute_env_vars(config)


def get_loader_class(loader_path: str) -> Type[BaseDataLoader]:
    """Import and return the loader class from the given module path."""
    try:
        module_path, class_name = loader_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        loader_class = getattr(module, class_name)
        if not issubclass(loader_class, BaseDataLoader):
            raise ValueError(f"{loader_path} is not a subclass of BaseDataLoader")
        return loader_class
    except (ImportError, AttributeError, ValueError) as e:
        raise click.BadParameter(f"Failed to import loader class: {str(e)}")


async def _load_data_direct(loader_class: Type[BaseDataLoader], config_data: Dict[str, Any]) -> dd.DataFrame:
    loader_instance = loader_class(**config_data)
    return await loader_instance.to_dask_dataframe()


async def _run_pipeline(config: Dict[str, Any]):
    pipeline = Pipeline(config)
    await pipeline.run()


@click.group()
def cli():
    """Data loader CLI interface for loading and processing data."""
    pass


@cli.command()
@click.option('--loader', '-l', required=True, help='Path to the loader class (e.g., j1.data_loaders.pubmed.PubMedDataLoader)')
@click.option('--config', '-c', help='Path to YAML/JSON config file')
@click.option('--output', '-o', help='Output path for the resulting Dask DataFrame')
@click.option('--shard-size', type=int, help='Number of documents per shard')
@click.option('--concurrency', type=int, help='Number of concurrent downloads')
@click.option('--article-limit', type=int, help='Maximum number of articles to process')
def load(loader: str, config: str, output: str, shard_size: int, concurrency: int, article_limit: int):
    """Load data using the specified loader.
    
    Examples:
        \b
        # Using command line arguments
        j1 load --loader j1.data_loaders.pubmed.PubMedDataLoader \\
                --shard-size 1000 \\
                --concurrency 5 \\
                --article-limit 1000 \\
                --output data.parquet
        
        # Using a config file
        j1 load --loader j1.data_loaders.pubmed.PubMedDataLoader \\
                --config config.yaml \\
                --output data.parquet
    """
    loader_class = get_loader_class(loader)
    
    config_data = {}
    if config:
        config_data = load_config(config)
    
    if shard_size is not None:
        config_data['shard_size'] = shard_size
    if concurrency is not None:
        config_data['concurrency'] = concurrency
    if article_limit is not None:
        config_data['article_limit'] = article_limit
    
    click.echo(f"Loading data using {loader}...")
    
    # Run the async operation
    df = asyncio.run(_load_data_direct(loader_class, config_data))
    
    if output:
        click.echo(f"Saving to {output}...")
        with ProgressBar():
            df.to_parquet(output)
        click.echo("Done!")
    else:
        click.echo("Data loaded successfully. Use --output to save the results.")


@cli.command(name="run-pipeline")
@click.option('--config', '-c', required=True, help='Path to the pipeline YAML/JSON config file')
def run_pipeline(config: str):
    """Run a data processing pipeline defined in a config file.

    Examples:
        j1 run-pipeline --config configs/pubmed_summary_pipeline.yaml
    """
    try:
        config_data = load_config(config)
        click.echo(f"Running pipeline defined in {config}...")
        asyncio.run(_run_pipeline(config_data))
    except Exception as e:
        click.echo(f"Error running pipeline: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Path to the evaluation configuration file')
@click.option('--output', type=click.Path(), help='Path to save the results CSV file')
def evaluate(config, output):
    """Run an evaluation using the specified configuration."""
    try:
        # Load configuration with environment variable substitution
        config_data = load_config(config)
        
        # Get benchmark type
        benchmark_type = config_data.get("benchmark_type")
        if not benchmark_type:
            raise ValueError("benchmark_type must be specified in the config")
        
        # Import appropriate evaluator
        if benchmark_type == "medqa":
            from .eval.medqa import MedQAEvaluator, MultiModelEvaluator
            evaluator_class = MedQAEvaluator
        elif benchmark_type == "pubmedqa":
            from .eval.pubmedqa import PubMedQAEvaluator
            from .eval.medqa import MultiModelEvaluator
            evaluator_class = PubMedQAEvaluator
        else:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")
        
        # Create evaluator
        evaluator = MultiModelEvaluator(config_data, evaluator_class)
        
        # Run evaluation
        print("Starting evaluation for all models...")
        results = asyncio.run(evaluator.evaluate_all())
        
        # Generate and print summary
        evaluator.print_summary(results)
        
        # Save results if output path is specified
        if output:
            # Generate comparative report
            comparative_report = evaluator.generate_comparative_report(results)
            comparative_report.to_csv(output, index=False)
            print(f"Results saved to {output}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise e


def main():
    cli(prog_name='j1')


if __name__ == '__main__':
    main() 