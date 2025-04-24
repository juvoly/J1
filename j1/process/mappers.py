import asyncio
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable

import dask.dataframe as dd
import pandas as pd
# No dask.delayed needed here as map_partitions handles async functions
from openai import AsyncOpenAI, OpenAIError

from j1.process.base import Processor

logger = logging.getLogger(__name__)


async def _apply_openai_prompt_partition(
    partition: pd.DataFrame,
    input_columns: List[str],
    output_columns: List[str],
    user_prompt_template: str,
    openai_model: str,
    openai_api_key: str,
    openai_kwargs: Dict[str, Any],
    max_concurrent_requests: int,
    extract_content_func: Callable[[str], Dict[str, Any]],
    get_system_prompt_func: Callable[[pd.Series], str],
    openai_base_url: Optional[str] = None,
) -> pd.DataFrame:
    """
    Asynchronously applies OpenAI prompt to each row of a pandas DataFrame partition.
    This function is intended to be used with dask's map_partitions.
    Limits concurrency using a semaphore.
    """
    if partition.empty:
        # For empty partitions, create a DataFrame with the output columns
        empty_data = {col: pd.Series(dtype="str") for col in output_columns}
        return pd.DataFrame(empty_data, index=partition.index)

    client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_base_url)
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def process_row(row: pd.Series):
        """Processes a single row asynchronously, respecting the semaphore."""
        try:
            prompt_data = {col: row[col] for col in input_columns}
            user_prompt = user_prompt_template.format(**prompt_data)
            system_prompt = get_system_prompt_func(row)

            async with semaphore:
                response = await client.chat.completions.create(
                    model=openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    **openai_kwargs,
                )
            content = response.choices[0].message.content

            extracted_content = content or "" # Return raw content (or empty string if None)
            return extracted_content
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error: OpenAI API - {e}"
        except KeyError as e:
            logger.error(f"Missing key '{e}' for formatting prompt")
            return "Error: Missing template key"
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return f"Error: Unexpected - {e}"

    tasks = [process_row(row) for _, row in partition.iterrows()]
    raw_results_list = await asyncio.gather(*tasks) # List of raw content strings or error messages

    # Create a new DataFrame with original data + new columns initialized
    result_df = partition.copy()
    for col in output_columns:
        result_df[col] = pd.Series(dtype="object", index=partition.index) # Use object for flexibility initially

    # Process results and populate output columns
    extracted_data_list = []
    for raw_content in raw_results_list:
        if isinstance(raw_content, str) and raw_content.startswith("Error:"):
             # Propagate errors to all output columns for the row
            extracted_data_list.append({col: raw_content for col in output_columns})
        else:
            try:
                extracted_dict = extract_content_func(raw_content)
                extracted_data_list.append(extracted_dict)
            except Exception as e:
                logger.error(f"Error during content extraction: {e}", exc_info=True)
                extracted_data_list.append({col: f"Error: Extraction failed - {e}" for col in output_columns})

    temp_extracted_df = pd.DataFrame(extracted_data_list, index=partition.index)
    result_df.update(temp_extracted_df) # Update result_df with extracted values
    return result_df


class PromptMapper(Processor, ABC):
    """
    A Processor that applies a generative AI prompt (via OpenAI API)
    to each row of a Dask DataFrame, storing the result in a new column.

    Uses dask's map_partitions with asynchronous processing and concurrency limiting
    for efficiency and rate limit management.
    """

    def __init__(
        self,
        input_columns: List[str],
        output_columns: List[str],
        user_prompt_template: str,
        system_prompt: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
        openai_kwargs: Optional[Dict[str, Any]] = None,
        max_concurrent_requests: int = 10,
        max_tokens: int = 1000,
        openai_base_url: Optional[str] = None,
    ):
        """Initialize the PromptMapper.
        Args:
            input_columns: List of input column names required for user_prompt_template.
            output_columns: List of new column names for the extracted OpenAI response parts.
            user_prompt_template: f-string template for user prompt. All placeholders
                                  (e.g., {col_name}) must exist in input_columns.
            system_prompt: Optional static system prompt. If None, generate_system_prompt will be used.
            openai_model: OpenAI model identifier.
            openai_api_key: OpenAI API key (or reads OPENAI_API_KEY env var).
            openai_kwargs: Additional kwargs for OpenAI API call.
            max_concurrent_requests: Maximum concurrent OpenAI requests per partition (default: 10).
            max_tokens: Maximum number of tokens to generate in the response (default: 1000).
            openai_base_url: Custom base URL for OpenAI API (e.g., for local servers).
        """
        if not input_columns:
            raise ValueError("input_columns list cannot be empty.")
        if not output_columns:
             raise ValueError("output_columns list cannot be empty.")
        if not user_prompt_template:
            raise ValueError("user_prompt_template cannot be empty.")
        if max_concurrent_requests <= 0:
             raise ValueError("max_concurrent_requests must be positive.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")

        self.input_columns = input_columns
        self.output_columns = output_columns
        self.user_prompt_template = user_prompt_template
        self.static_system_prompt = system_prompt
        self.openai_model = openai_model
        self.openai_kwargs = openai_kwargs or {}
        self.openai_kwargs['max_tokens'] = max_tokens
        self.max_concurrent_requests = max_concurrent_requests
        self.openai_base_url = openai_base_url

        # Resolve API Key
        resolved_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "OpenAI API key not provided and 'OPENAI_API_KEY' env var not set."
            )
        self.openai_api_key = resolved_api_key

        # Refined check: ensure all template placeholders are in input_columns
        try:
             placeholders = set(re.findall(r"\{(\w+)\}", self.user_prompt_template))
             available_inputs = set(self.input_columns)
             unknown_placeholders = placeholders - available_inputs
             if unknown_placeholders:
                 # Raise specifically if template uses columns not provided
                 raise KeyError(f"Template contains placeholders not found in input_columns: {list(unknown_placeholders)}")
        except KeyError as e:
             raise ValueError(str(e)) from e # Raise ValueError for consistency
        except Exception as e:
             raise ValueError(f"Error validating user_prompt_template syntax: {e}") from e

    def get_system_prompt(self, row: pd.Series) -> str:
        """
        Gets the system prompt for a given row. If a static system prompt was provided,
        returns that. Otherwise, calls generate_system_prompt.
        """
        if self.static_system_prompt is not None:
            return self.static_system_prompt
        return self.generate_system_prompt(row)

    def generate_system_prompt(self, row: pd.Series) -> str:
        """
        Generates the system prompt based on the input row data.
        Subclasses must implement this method to define how the system prompt
        is generated for each row when no static system prompt is provided.

        Args:
            row: A pandas Series containing the input data for the current row.

        Returns:
            A string containing the system prompt.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def extract_content(self, response_content: str) -> Dict[str, Any]:
        """
        Extracts structured data from the raw OpenAI response string.
        Subclasses must implement this method to define how the response
        maps to the defined `output_columns`.

        Returns:
            A dictionary where keys are subset of self.output_columns and values are the extracted data.
        """
        pass

    def process(self, df: dd.DataFrame) -> dd.DataFrame:
        """
        Applies the OpenAI prompt processing to the input Dask DataFrame.
        """
        # Validate input columns exist
        missing_cols = [col for col in self.input_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Input DataFrame missing required columns: {missing_cols}. "
                f"Available: {list(df.columns)}"
            )

        # Define the metadata for the output DataFrame
        meta = df._meta.copy() if hasattr(df, '_meta') else df.head(0)
        for col in self.output_columns:
             meta[col] = pd.Series(dtype="str") # Use string as default meta type

        def sync_wrapper(partition):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    _apply_openai_prompt_partition(
                        partition=partition,
                        input_columns=self.input_columns,
                        output_columns=self.output_columns,
                        user_prompt_template=self.user_prompt_template,
                        openai_model=self.openai_model,
                        openai_api_key=self.openai_api_key,
                        openai_kwargs=self.openai_kwargs,
                        max_concurrent_requests=self.max_concurrent_requests,
                        extract_content_func=self.extract_content,
                        get_system_prompt_func=self.get_system_prompt,
                        openai_base_url=self.openai_base_url,
                    )
                )
                return result
            finally:
                loop.close()

        # Apply the partition processing function
        result_df = df.map_partitions(
            sync_wrapper,
            meta=meta,
            enforce_metadata=True,
        )

        # Ensure final output columns have consistent dtype, converting object if possible
        for col in self.output_columns:
             if col in result_df.columns: # Check if column exists (could fail in empty partitions?)
                 result_df[col] = result_df[col].astype(str)

        return result_df


class SimplePromptMapper(PromptMapper):
    """
    A concrete PromptMapper that takes raw text input from one or more columns,
    formats it into a prompt, calls the OpenAI API, and stores the *entire*
    response content in a single designated output column without further parsing.
    """
    def __init__(
        self,
        input_columns: List[str],
        output_column: str,
        system_prompt: str,
        user_prompt_template: str,
        openai_model: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
        openai_kwargs: Optional[Dict[str, Any]] = None,
        max_concurrent_requests: int = 10,
        max_tokens: int = 1000,
        openai_base_url: Optional[str] = None,
    ):
        super().__init__(
            input_columns=input_columns,
            output_columns=[output_column],
            user_prompt_template=user_prompt_template,
            system_prompt=system_prompt,
            openai_model=openai_model,
            openai_api_key=openai_api_key,
            openai_kwargs=openai_kwargs,
            max_concurrent_requests=max_concurrent_requests,
            max_tokens=max_tokens,
            openai_base_url=openai_base_url,
        )

    def extract_content(self, response_content: str) -> Dict[str, Any]:
        return {self.output_columns[0]: response_content.strip()} 