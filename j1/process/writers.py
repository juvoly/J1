import json
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import dask.dataframe as dd
import pandas as pd
from jinja2 import Template


class DataWriter(ABC):
    """Base class for writing data to files in parallel using Dask."""

    def __init__(self, output_path: str):
        """Initialize the data writer.
        
        Args:
            output_path: Path to the output file
        """
        self.output_path = output_path
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    @abstractmethod
    def _convert_to_jsonl(self, row: pd.Series) -> str:
        """Convert a row to a JSONL string.
        
        Args:
            row: A pandas Series containing the row data
            
        Returns:
            A JSONL string representing the row
        """
        pass

    def write(self, df: dd.DataFrame):
        """Write the dataframe to a JSONL file in parallel.
        
        Args:
            df: Input dask dataframe
        """
        def process_partition(partition: pd.DataFrame) -> List[str]:
            """Process a partition and return JSONL strings.
            
            Args:
                partition: A pandas DataFrame partition
                
            Returns:
                List of JSONL strings
            """
            jsonl_strings = []
            for _, row in partition.iterrows():
                try:
                    jsonl_str = self._convert_to_jsonl(row)
                    jsonl_strings.append(jsonl_str)
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            return jsonl_strings

        # Process all partitions and collect results
        all_jsonl = df.map_partitions(
            process_partition,
            meta=pd.Series(dtype='object'),  # Metadata for the list of strings
            enforce_metadata=False
        ).compute()

        # Flatten the list of lists
        all_jsonl = [item for sublist in all_jsonl for item in sublist]

        # Clear the output file if it exists
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

        # Write all JSONL strings to the file
        with open(self.output_path, 'w') as f:
            for jsonl_str in all_jsonl:
                f.write(jsonl_str + '\n')


class QASFTDataWriter(DataWriter):
    """Writer for SFT samples in JSONL format, specifically for QA pairs."""

    def __init__(
        self,
        output_path: str,
        question_column: str = "question",
        answer_column: str = "answer",
        validation_split: float = 0.1,
        seed: int = 42,
    ):
        """Initialize the QA SFT data writer.
        
        Args:
            output_path: Path to the output JSONL file (without _train or _val suffix)
            question_column: Name of the column containing the question
            answer_column: Name of the column containing the answer
            validation_split: Fraction of data to use for validation (default: 0.1)
            seed: Random seed for reproducibility (default: 42)
        """
        super().__init__(output_path)
        self.question_column = question_column
        self.answer_column = answer_column
        self.validation_split = validation_split
        self.seed = seed
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Create paths for train and validation files
        base_path, ext = os.path.splitext(output_path)
        self.train_path = f"{base_path}_train{ext}"
        self.val_path = f"{base_path}_val{ext}"

    def _convert_to_jsonl(self, row: pd.Series) -> str:
        """Convert a QA pair to a JSONL string with conversation format.
        
        Args:
            row: A pandas Series containing the question and answer
            
        Returns:
            A JSONL string with the conversation format
        """
        # Handle NA values by converting them to empty strings
        question = str(row[self.question_column]) if pd.notna(row[self.question_column]) else ""
        answer = str(row[self.answer_column]) if pd.notna(row[self.answer_column]) else ""
        
        # Skip empty questions or answers
        if not question or not answer:
            raise ValueError("Empty question or answer")
        
        # Create the conversation structure with two messages
        conversation = {
            "conversations": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }
        
        # Convert to JSONL string
        return json.dumps(conversation)

    def write(self, df: dd.DataFrame):
        """Write the dataframe to train and validation JSONL files.
        
        Args:
            df: Input dask dataframe
        """
        def process_partition(partition: pd.DataFrame) -> Tuple[List[str], List[str]]:
            """Process a partition and return train and validation JSONL strings.
            
            Args:
                partition: A pandas DataFrame partition
                
            Returns:
                Tuple of (train_jsonl_strings, val_jsonl_strings)
            """
            train_jsonl = []
            val_jsonl = []
            
            for _, row in partition.iterrows():
                try:
                    jsonl_str = self._convert_to_jsonl(row)
                    # Randomly assign to train or validation
                    if random.random() < self.validation_split:
                        val_jsonl.append(jsonl_str)
                    else:
                        train_jsonl.append(jsonl_str)
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
                    
            return train_jsonl, val_jsonl

        # Process all partitions and collect results
        all_results = df.map_partitions(
            process_partition,
            meta=pd.Series(dtype='object'),  # Metadata for the tuple of lists
            enforce_metadata=False
        ).compute()

        # Separate and flatten train and validation results
        train_jsonl = [item for sublist in [r[0] for r in all_results] for item in sublist]
        val_jsonl = [item for sublist in [r[1] for r in all_results] for item in sublist]

        # Clear existing files
        for path in [self.train_path, self.val_path]:
            if os.path.exists(path):
                os.remove(path)

        # Write train and validation files
        with open(self.train_path, 'w') as train_f, open(self.val_path, 'w') as val_f:
            for jsonl_str in train_jsonl:
                train_f.write(jsonl_str + '\n')
            for jsonl_str in val_jsonl:
                val_f.write(jsonl_str + '\n')
        
        # Print statistics
        train_count = len(train_jsonl)
        val_count = len(val_jsonl)
        total = train_count + val_count
        print(f"Split complete: {train_count} train samples ({train_count/total:.1%}), "
              f"{val_count} validation samples ({val_count/total:.1%})") 