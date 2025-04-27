import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Type
import json
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import time
import logging
from tabulate import tabulate

import aiohttp
import pandas as pd
from tqdm import tqdm
from .base import QAEvaluator, QAPrompt

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Token pricing per 1M tokens
TOKEN_PRICING = {
    # OpenAI Models
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4.5-preview": {"input": 75.00, "output": 150.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-pro": {"input": 150.00, "output": 600.00},
    "o3": {"input": 10.00, "output": 40.00},
    "o4-mini": {"input": 1.10, "output": 4.40},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o1-mini": {"input": 1.10, "output": 4.40},
    
    # Anthropic Models
    "claude-3-7-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},
    "claude-3-7-haiku-20240307": {"input": 0.25, "output": 1.25},
    "claude-2.1": {"input": 8.00, "output": 24.00},
    "claude-2.0": {"input": 8.00, "output": 24.00},
    "claude-instant-1.2": {"input": 0.80, "output": 2.40},

    # Google Gemini Models
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "J1": {"input": 1.25, "output": 1.25},
}

def calculate_token_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate the cost of tokens used in dollars."""
    if model not in TOKEN_PRICING:
        logger.warning(f"No pricing information available for model {model}")
        return 0.0
    
    pricing = TOKEN_PRICING[model]
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost

@dataclass
class MedQAPrompt:
    """Prompt for MedQA evaluation"""
    question: str
    options: Dict[str, str]
    use_chain_of_thought: bool = False
    custom_prompt: Optional[str] = None
    
    def format_prompt(self) -> str:
        if self.custom_prompt is not None:
            return self.custom_prompt.format(
                question=self.question,
                options=self.options
            )
            
        options_text = "\n".join([f"{key}) {value}" for key, value in self.options.items()])
        
        if self.use_chain_of_thought:
            return f"""You are a medical expert. Please answer the following question by selecting the most appropriate option.

Question: {self.question}

Options:
{options_text}

Please follow these steps:
1. First, explain your reasoning step by step
2. Then, provide your final answer in the following format:
Answer: <letter>

For example:
1. The patient is experiencing symptoms X and Y
2. Based on medical knowledge Z
3. The most appropriate action is C
Answer: C"""
        else:
            return f"""{self.question}\n{options_text}"""

class MedQAEvaluator(QAEvaluator):
    def create_prompt(self, item: Dict) -> MedQAPrompt:
        return MedQAPrompt(
            question=item["question"],
            options=item["options"],
            use_chain_of_thought=self.use_chain_of_thought,
            custom_prompt=self.custom_prompt
        )
    
    def extract_answer(self, response: str) -> str:
        lines = response.strip().split("\n")
        for line in lines:
            line = line.replace("*", "").lower()
            if line.strip().startswith("answer:"):
                return line.strip().split(":")[1].strip().upper()
            if line.strip().startswith("the best answer is "):
                return line.strip().split("is ")[1].strip().replace(".", "").upper()
        return ""
    
    def is_correct(self, extracted_answer: str, correct_answer: str) -> bool:
        return extracted_answer == correct_answer

class MultiModelEvaluator:
    def __init__(self, config: Dict, evaluator_class: Type[QAEvaluator]):
        self.config = config
        self.evaluator_class = evaluator_class
        self.evaluators = [
            evaluator_class(
                label=model["label"],
                model=model["model"],
                api_base=model["api_base"],
                api_key=model["api_key"],
                max_concurrent_requests=model.get("max_concurrent_requests", 5),
                model_params=model.get("params", {}),
                use_chain_of_thought=model.get("use_chain_of_thought", False),
                custom_prompt=model.get("custom_prompt")
            )
            for model in config["models"]
        ]
    
    async def evaluate_all(self) -> Dict[str, pd.DataFrame]:
        """Evaluate all models and return a dictionary of results."""
        results = {}
        limit = self.config.get("limit")
        
        if limit is not None:
            print(f"Limiting evaluation to first {limit} items")
        
        # Run evaluations concurrently for all models
        tasks = [
            evaluator.evaluate_dataset(self.config["dataset_path"], limit=limit)
            for evaluator in self.evaluators
        ]
        
        all_results = await asyncio.gather(*tasks)
        
        # Generate reports for each model
        for evaluator, model_results in zip(self.evaluators, all_results):
            results[evaluator.label] = evaluator.generate_report(model_results)
        
        return results
    
    def generate_comparative_report(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate a comparative report across all models."""
        # Combine all model reports
        combined_df = pd.concat(results.values(), ignore_index=True)
        
        # Sort by accuracy
        combined_df = combined_df.sort_values("accuracy", ascending=False)
        
        return combined_df
    
    def print_summary(self, results: Dict[str, pd.DataFrame]):
        """Print a detailed summary of the evaluation results."""
        print("\n" + "="*80)
        print("MEDICAL QA EVALUATION SUMMARY")
        print("="*80)
        
        # Print evaluation details
        print(f"\nEvaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Dataset: {self.config['dataset_path']}")
        if self.config.get('limit'):
            print(f"Items Evaluated: {self.config['limit']} (limited)")
        else:
            print("Items Evaluated: All available items")
        
        # Check for any errors across all models
        total_errors = sum(df.iloc[0]['error_count'] for df in results.values())
        if total_errors > 0:
            print(f"\nWARNING: {total_errors} total evaluation errors across all models")
            print("See warning messages above for details")
        
        # Prepare table data
        table_data = []
        for label, df in results.items():
            model_results = df.iloc[0]
            table_data.append({
                "Rank": len(table_data) + 1,
                "Model": label,
                "Accuracy": f"{model_results['accuracy']:.2%}",
                "Correct": f"{model_results['correct_answers']}/{model_results['total_questions']}",
                "Errors": f"{model_results['error_count']}",
                "CoT": "✓" if model_results['use_chain_of_thought'] else "✗",
                "Total Tokens": f"{model_results['total_tokens']:,}",
                "Total Cost": f"${model_results['total_cost']:.4f}",
                "Score": model_results['accuracy']  # For sorting
            })
        
        # Sort by accuracy
        table_data.sort(key=lambda x: x["Score"], reverse=True)
        for i, row in enumerate(table_data):
            row["Rank"] = i + 1
        
        # Print performance table
        print("\n" + "-"*80)
        print("PERFORMANCE SUMMARY")
        print("-"*80)
        print(tabulate(
            [{
                "Rank": row["Rank"],
                "Model": row["Model"],
                "Accuracy": row["Accuracy"],
                "Correct": row["Correct"],
                "Errors": row["Errors"],
                "CoT": row["CoT"],
                "Total Tokens": row["Total Tokens"],
                "Total Cost": row["Total Cost"]
            } for row in table_data],
            headers="keys",
            tablefmt="grid"
        ))
        
        # Print cost efficiency table
        print("\n" + "-"*80)
        print("COST EFFICIENCY")
        print("-"*80)
        print(tabulate(
            [{
                "Rank": row["Rank"],
                "Model": row["Model"],
                "Total Cost": row["Total Cost"],
                "Cost/Q": f"${results[row['Model']].iloc[0]['cost_per_question']:.4f}",
                "Input Tokens": f"{results[row['Model']].iloc[0]['total_prompt_tokens']:,}",
                "Output Tokens": f"{results[row['Model']].iloc[0]['total_completion_tokens']:,}",
                "Tokens/Q": f"{results[row['Model']].iloc[0]['total_tokens'] / results[row['Model']].iloc[0]['total_questions']:.0f}"
            } for row in table_data],
            headers="keys",
            tablefmt="grid"
        ))
        
        # Print pricing details table
        print("\n" + "-"*80)
        print("PRICING DETAILS")
        print("-"*80)
        print(tabulate(
            [{
                "Rank": row["Rank"],
                "Model": row["Model"],
                "Input Price": f"${TOKEN_PRICING[results[row['Model']].iloc[0]['model']]['input']}/1M",
                "Output Price": f"${TOKEN_PRICING[results[row['Model']].iloc[0]['model']]['output']}/1M",
                "Input Cost": f"${(results[row['Model']].iloc[0]['total_prompt_tokens'] / 1_000_000) * TOKEN_PRICING[results[row['Model']].iloc[0]['model']]['input']:.4f}",
                "Output Cost": f"${(results[row['Model']].iloc[0]['total_completion_tokens'] / 1_000_000) * TOKEN_PRICING[results[row['Model']].iloc[0]['model']]['output']:.4f}",
                "Total Cost": row["Total Cost"]
            } for row in table_data],
            headers="keys",
            tablefmt="grid"
        ))
        
        # Print detailed breakdowns
        print("\n" + "-"*80)
        print("DETAILED BREAKDOWNS")
        print("-"*80)
        
        for label, df in results.items():
            model_results = df.iloc[0]
            if 'meta_stats' in model_results and isinstance(model_results['meta_stats'], dict):
                print(f"\n{label} - Breakdown by Category:")
                category_data = []
                for category, stats in model_results['meta_stats'].items():
                    try:
                        accuracy = stats['sum'] / stats['count']
                        category_data.append({
                            "Category": category,
                            "Accuracy": f"{accuracy:.2%}",
                            "Correct": f"{stats['sum']}/{stats['count']}"
                        })
                    except (KeyError, TypeError, ZeroDivisionError):
                        # Skip invalid or missing stats
                        continue
                
                if category_data:
                    print(tabulate(category_data, headers="keys", tablefmt="grid"))
                else:
                    print("No valid category breakdowns available")
        
        print("\n" + "="*80)
