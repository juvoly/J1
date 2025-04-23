import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import dask.dataframe as dd
import pandas as pd
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
import re

# Make sure the test can find the modules
import sys
from pathlib import Path

from j1.pipeline.mappers import PromptMapper


# --- Fixtures ---

@pytest.fixture
def sample_pandas_df():
    """Provides a sample Pandas DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'text': ['This is the first document.', 'Here is the second one.', 'And the third.'],
        'extra_data': ['A', 'B', 'C']
    })

@pytest.fixture
def sample_dask_df(sample_pandas_df):
    """Provides a sample Dask DataFrame with 2 partitions."""
    return dd.from_pandas(sample_pandas_df, npartitions=2)

@pytest.fixture
def mock_openai_response():
    """Provides a mock successful OpenAI API response."""
    # Create a mock ChatCompletionMessage
    mock_message = ChatCompletionMessage(
        content=" MOCK RESPONSE ", # Note the leading/trailing whitespace
        role="assistant"
    )
    # Create a mock Choice using the mock message
    mock_choice = Choice(
        finish_reason="stop",
        index=0,
        message=mock_message
    )
    # Create the mock ChatCompletion response
    mock_completion = ChatCompletion(
        id="chatcmpl-mockid",
        choices=[mock_choice],
        created=1677652288, # Example timestamp
        model="gpt-4o-mini",
        object="chat.completion",
        system_fingerprint="fp_mock", # Example fingerprint
        usage=None # Usage details can be mocked if needed
    )
    return mock_completion

@pytest.fixture
def mock_openai_error_response():
    """Provides a mock failed OpenAI API response (simulates API error)."""
    # In recent openai versions, errors are raised, not returned in response body usually.
    # We'll mock the client's create method to raise an error instead.
    from openai import APIError
    return APIError("Mock API Error", request=None, body=None)

# --- Test Cases ---

@pytest.mark.asyncio
@patch('j1.pipeline.mappers.AsyncOpenAI', new_callable=MagicMock) # Patch the class
async def test_prompt_mapper_success(mock_async_openai_class, sample_dask_df, mock_openai_response):
    """Tests successful processing using PromptMapper."""
    # Configure the mock instance that will be created
    mock_client_instance = AsyncMock()
    mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_openai_response)
    # Make the class return our configured instance when called
    mock_async_openai_class.return_value = mock_client_instance

    mapper = PromptMapper(
        input_columns=['text'],
        output_column='summary',
        system_prompt='Summarize the following text.',
        user_prompt_template='Text: {text}',
        openai_api_key='fake_key' # Provide a fake key for initialization
    )

    result_ddf = mapper.process(sample_dask_df)
    result_pdf = result_ddf.compute()

    # Assertions
    assert 'summary' in result_pdf.columns
    assert len(result_pdf) == 3
    # Check that the mock response content (stripped) is in the output column
    pd.testing.assert_series_equal(
        result_pdf['summary'],
        pd.Series(["MOCK RESPONSE"] * 3, name='summary'),
        check_dtype=True # Ensure dtype is also compared (should be object/string)
    )

    # Check if the mock was called correctly for each row
    assert mock_client_instance.chat.completions.create.await_count == 3
    
    # Get all the user prompts that were sent
    user_prompts = [
        call[1]['messages'][1]['content']
        for call in mock_client_instance.chat.completions.create.await_args_list
    ]
    
    # Check that all expected prompts were sent (order doesn't matter)
    expected_prompts = {
        'Text: This is the first document.',
        'Text: Here is the second one.',
        'Text: And the third.'
    }
    assert set(user_prompts) == expected_prompts
    
    # Check the first call's other parameters
    first_call_args, first_call_kwargs = mock_client_instance.chat.completions.create.await_args_list[0]
    assert first_call_kwargs['model'] == 'gpt-4o-mini' # Default model
    assert first_call_kwargs['messages'][0]['role'] == 'system'
    assert first_call_kwargs['messages'][0]['content'] == 'Summarize the following text.'


@pytest.mark.asyncio
@patch('j1.pipeline.mappers.AsyncOpenAI', new_callable=MagicMock)
async def test_prompt_mapper_empty_dataframe(mock_async_openai_class, sample_dask_df):
    """Tests processing an empty DataFrame."""
    mock_client_instance = AsyncMock()
    mock_async_openai_class.return_value = mock_client_instance

    # Create an empty Dask DataFrame with the same structure
    empty_df = dd.from_pandas(pd.DataFrame(columns=sample_dask_df.columns), npartitions=1)

    mapper = PromptMapper(
        input_columns=['text'],
        output_column='summary',
        system_prompt='Summarize.',
        user_prompt_template='{text}',
        openai_api_key='fake_key'
    )

    result_ddf = mapper.process(empty_df)
    result_pdf = result_ddf.compute()

    assert 'summary' in result_pdf.columns
    assert len(result_pdf) == 0
    assert result_pdf['summary'].dtype == 'object' # Dask often defaults to object for strings

    # Ensure the API was not called
    mock_client_instance.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
@patch('j1.pipeline.mappers.AsyncOpenAI', new_callable=MagicMock)
async def test_prompt_mapper_openai_error(mock_async_openai_class, sample_dask_df, mock_openai_error_response):
    """Tests handling of OpenAI API errors during processing."""
    mock_client_instance = AsyncMock()
    # Configure the mock to raise the APIError
    mock_client_instance.chat.completions.create = AsyncMock(side_effect=mock_openai_error_response)
    mock_async_openai_class.return_value = mock_client_instance

    mapper = PromptMapper(
        input_columns=['text'],
        output_column='summary',
        system_prompt='Summarize.',
        user_prompt_template='{text}',
        openai_api_key='fake_key'
    )

    result_ddf = mapper.process(sample_dask_df)
    result_pdf = result_ddf.compute()

    assert 'summary' in result_pdf.columns
    # Check that the error message is stored in the output column
    expected_error_prefix = "Error: OpenAI API - Mock API Error"
    assert result_pdf['summary'].iloc[0].startswith(expected_error_prefix)
    assert result_pdf['summary'].iloc[1].startswith(expected_error_prefix)
    assert result_pdf['summary'].iloc[2].startswith(expected_error_prefix)


@pytest.mark.asyncio
@patch('j1.pipeline.mappers.AsyncOpenAI', new_callable=MagicMock)
async def test_prompt_mapper_init_key_error(mock_async_openai_class, sample_dask_df):
    """Tests initialization failure when template uses unknown placeholder."""
    # Template expects '{content}' but input_columns only provides 'text'
    expected_error_msg = "Template contains placeholders not found in input_columns: ['content']"
    with pytest.raises(ValueError, match=re.escape(expected_error_msg)):
        PromptMapper(
            input_columns=['text'], # Only provide 'text'
            output_column='summary',
            system_prompt='Summarize.',
            user_prompt_template='{content}', # Use a non-existent key
            openai_api_key='fake_key'
        )


def test_prompt_mapper_init_missing_api_key():
    """Tests initialization failure when API key is missing."""
    with patch.dict(os.environ, {}, clear=True): # Ensure env var is not set
         with pytest.raises(ValueError, match="OpenAI API key not provided and 'OPENAI_API_KEY' env var not set"):
            PromptMapper(
                input_columns=['text'],
                output_column='summary',
                system_prompt='System prompt',
                user_prompt_template='User: {text}',
                openai_api_key=None # Explicitly pass None
            )

def test_prompt_mapper_init_provided_api_key():
    """Tests successful initialization with a provided API key, ignoring env var."""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'env_var_key'}, clear=True):
        try:
            mapper = PromptMapper(
                input_columns=['text'],
                output_column='summary',
                system_prompt='System prompt',
                user_prompt_template='User: {text}',
                openai_api_key='provided_key' # Provide key directly
            )
            assert mapper.openai_api_key == 'provided_key'
        except ValueError:
            pytest.fail("Initialization failed unexpectedly when API key was provided.")

def test_prompt_mapper_init_env_var_api_key():
    """Tests successful initialization using API key from environment variable."""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'env_var_key'}, clear=True):
        try:
            mapper = PromptMapper(
                input_columns=['text'],
                output_column='summary',
                system_prompt='System prompt',
                user_prompt_template='User: {text}',
                openai_api_key=None # Do not provide directly
            )
            assert mapper.openai_api_key == 'env_var_key'
        except ValueError:
            pytest.fail("Initialization failed unexpectedly when API key was in env var.")


def test_prompt_mapper_init_invalid_template():
    """Tests initialization failure with a template missing required placeholders."""
    # Template uses {invalid_col} which is not in input_columns
    expected_error_msg = "Template contains placeholders not found in input_columns: ['invalid_col']"
    with pytest.raises(ValueError, match=re.escape(expected_error_msg)):
        PromptMapper(
            input_columns=['text', 'extra_data'],
            output_column='summary',
            system_prompt='System prompt',
            user_prompt_template='User: {invalid_col} only', # Use placeholder not in inputs
            openai_api_key='fake_key'
        )

def test_prompt_mapper_init_valid_template_subset_cols():
    """Tests successful initialization when template uses a subset of input_columns."""
    try:
        PromptMapper(
            input_columns=['text', 'extra_data'], # Provide more cols than template uses
            output_column='summary',
            system_prompt='System prompt',
            user_prompt_template='User: {extra_data} only', # Only uses extra_data
            openai_api_key='fake_key'
        )
    except ValueError:
        pytest.fail("Initialization failed unexpectedly for valid template using subset of input columns.")

def test_prompt_mapper_init_empty_inputs():
    """Tests initialization failures with empty essential arguments."""
    with pytest.raises(ValueError, match="input_columns list cannot be empty"):
         PromptMapper([], 'out', 'sys', 'user', openai_api_key='fake')
    with pytest.raises(ValueError, match="output_column cannot be empty"):
         PromptMapper(['in'], '', 'sys', 'user', openai_api_key='fake')
    # System prompt being empty should only warn, not raise error
    # with pytest.raises(ValueError, match="system_prompt cannot be empty"):
    #     PromptMapper(['in'], 'out', '', 'user', openai_api_key='fake')
    with pytest.raises(ValueError, match="user_prompt_template cannot be empty"):
         PromptMapper(['in'], 'out', 'sys', '', openai_api_key='fake')


@pytest.mark.asyncio
@patch('j1.pipeline.mappers.AsyncOpenAI', new_callable=MagicMock)
async def test_prompt_mapper_process_missing_input_column(mock_async_openai_class, sample_dask_df):
    """Tests error handling when input_columns are missing from the DataFrame during process()."""
    mock_client_instance = AsyncMock()
    mock_async_openai_class.return_value = mock_client_instance

    mapper = PromptMapper(
        input_columns=['non_existent_column'], # This column is not in sample_dask_df
        output_column='summary',
        system_prompt='Summarize.',
        user_prompt_template='{non_existent_column}',
        openai_api_key='fake_key'
    )

    with pytest.raises(ValueError, match=re.escape("Input DataFrame missing required columns: ['non_existent_column']. Available: ['id', 'text', 'extra_data']")):
        mapper.process(sample_dask_df)

    mock_client_instance.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
@patch('j1.pipeline.mappers.AsyncOpenAI', new_callable=MagicMock)
async def test_prompt_mapper_multiple_input_columns(mock_async_openai_class, sample_dask_df, mock_openai_response):
    """Tests using multiple input columns in the prompt template."""
    mock_client_instance = AsyncMock()
    mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_openai_response)
    mock_async_openai_class.return_value = mock_client_instance

    mapper = PromptMapper(
        input_columns=['text', 'extra_data'], # Use two columns
        output_column='processed',
        system_prompt='Process the following.',
        user_prompt_template='Text: {text}\nExtra: {extra_data}', # Template uses both, corrected newline
        openai_api_key='fake_key'
    )

    result_ddf = mapper.process(sample_dask_df)
    result_pdf = result_ddf.compute()

    assert 'processed' in result_pdf.columns
    assert len(result_pdf) == 3
    assert (result_pdf['processed'] == "MOCK RESPONSE").all()

    assert mock_client_instance.chat.completions.create.await_count == 3
    
    # Get all the user prompts that were sent
    user_prompts = [
        call[1]['messages'][1]['content']
        for call in mock_client_instance.chat.completions.create.await_args_list
    ]
    
    # Check that all expected prompts were sent (order doesn't matter)
    expected_prompts = {
        'Text: This is the first document.\nExtra: A',
        'Text: Here is the second one.\nExtra: B',
        'Text: And the third.\nExtra: C'
    }
    assert set(user_prompts) == expected_prompts


@pytest.mark.asyncio
@patch('j1.pipeline.mappers.AsyncOpenAI', new_callable=MagicMock)
async def test_prompt_mapper_with_openai_kwargs(mock_async_openai_class, sample_dask_df, mock_openai_response):
    """Tests passing additional kwargs to the OpenAI API call."""
    mock_client_instance = AsyncMock()
    mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_openai_response)
    mock_async_openai_class.return_value = mock_client_instance

    openai_custom_kwargs = {'temperature': 0.5, 'max_tokens': 50}

    mapper = PromptMapper(
        input_columns=['text'],
        output_column='summary',
        system_prompt='Summarize.',
        user_prompt_template='{text}',
        openai_api_key='fake_key',
        openai_kwargs=openai_custom_kwargs # Pass extra kwargs
    )

    result_ddf = mapper.process(sample_dask_df)
    result_ddf.compute() # Trigger the computation

    assert mock_client_instance.chat.completions.create.await_count == 3
    # Check that the custom kwargs were passed in the API call
    first_call_args, first_call_kwargs = mock_client_instance.chat.completions.create.await_args_list[0]
    assert first_call_kwargs['temperature'] == 0.5
    assert first_call_kwargs['max_tokens'] == 50 