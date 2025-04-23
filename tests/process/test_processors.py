import pytest
import dask.dataframe as dd
import pandas as pd
import numpy as np

from j1.process import (
    CharacterCountFilter,
    SubstringProcessor,
    FieldProcessor
)


@pytest.fixture
def sample_dataframe():
    """Create a sample dask dataframe for testing."""
    data = {
        'text': [
            'Short text',
            'This is a medium length text',
            'This is a very long text that exceeds the maximum character limit',
            'Another medium text',
            'Just right'
        ],
        'id': [1, 2, 3, 4, 5],
        'extra': ['a', 'b', 'c', 'd', 'e']
    }
    pdf = pd.DataFrame(data)
    return dd.from_pandas(pdf, npartitions=2)


def test_character_count_filter(sample_dataframe):
    """Test the CharacterCountFilter processor."""
    # Test with min characters
    filter_min = CharacterCountFilter(text_key='text', min_characters=11)
    result = filter_min.process(sample_dataframe)
    pdf = result.compute()
    assert len(pdf) == 3  # Should exclude 'Short text' (len=10) and 'Just right' (len=10)
    assert 'Short text' not in pdf['text'].values
    
    # Test with max characters
    filter_max = CharacterCountFilter(text_key='text', max_characters=30)
    result = filter_max.process(sample_dataframe)
    pdf = result.compute()
    assert len(pdf) == 4  # Should exclude the very long text (len=68)
    assert not any(len(text) > 30 for text in pdf['text'].values)
    
    # Test with both min and max
    filter_both = CharacterCountFilter(
        text_key='text',
        min_characters=10,
        max_characters=30
    )
    result = filter_both.process(sample_dataframe)
    pdf = result.compute()
    assert len(pdf) == 4  # Should keep texts with length >= 10 and <= 30
    assert all(10 <= len(text) <= 30 for text in pdf['text'].values)
    
    # Test with invalid text key
    with pytest.raises(ValueError):
        filter_invalid = CharacterCountFilter(text_key='nonexistent')
        filter_invalid.process(sample_dataframe)


def test_substring_processor(sample_dataframe):
    """Test the SubstringProcessor."""
    # Test with start and end indices
    processor_slice = SubstringProcessor(
        text_key='text',
        start=0,
        end=10
    )
    result = processor_slice.process(sample_dataframe)
    pdf = result.compute()
    assert all(len(text) <= 10 for text in pdf['text'].values)
    assert pdf['text'].iloc[0] == 'Short text'  # Unchanged as it's already short
    
    # Test with pattern
    processor_pattern = SubstringProcessor(
        text_key='text',
        pattern=r'(.*text)$'  # Match any text that ends with 'text' with a capture group
    )
    result = processor_pattern.process(sample_dataframe)
    pdf = result.compute()
    assert all(text.endswith('text') for text in pdf['text'].values)
    
    # Test with invalid text key
    with pytest.raises(ValueError):
        processor_invalid = SubstringProcessor(text_key='nonexistent')
        processor_invalid.process(sample_dataframe)


def test_field_processor(sample_dataframe):
    """Test the FieldProcessor."""
    # Test dropping single field
    processor_single = FieldProcessor('extra')
    result = processor_single.process(sample_dataframe)
    pdf = result.compute()
    assert 'extra' not in pdf.columns
    assert 'text' in pdf.columns
    assert 'id' in pdf.columns
    
    # Test dropping multiple fields
    processor_multiple = FieldProcessor(['extra', 'id'])
    result = processor_multiple.process(sample_dataframe)
    pdf = result.compute()
    assert 'extra' not in pdf.columns
    assert 'id' not in pdf.columns
    assert 'text' in pdf.columns
    
    # Test dropping non-existent field (should not raise error)
    processor_nonexistent = FieldProcessor('nonexistent')
    result = processor_nonexistent.process(sample_dataframe)
    pdf = result.compute()
    assert len(pdf.columns) == len(sample_dataframe.columns)
    
    # Test with string input
    processor_string = FieldProcessor('extra')
    result = processor_string.process(sample_dataframe)
    pdf = result.compute()
    assert 'extra' not in pdf.columns 