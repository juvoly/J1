from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json
import logging
import time
import asyncio
import aiohttp
from tqdm import tqdm
import pandas as pd

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
class QAPrompt:
    """Base class for QA prompts"""
    question: str
    custom_prompt: Optional[str] = None
    
    @abstractmethod
    def format_prompt(self) -> str:
        """Format the prompt for the model"""
        pass

class QAEvaluator(ABC):
    """Base class for QA evaluation"""
    
    def __init__(
        self,
        label: str,
        model: str,
        api_base: str,
        api_key: str,
        max_concurrent_requests: int = 5,
        model_params: Optional[Dict] = None,
        use_chain_of_thought: bool = False,
        custom_prompt: Optional[str] = None
    ):
        self.label = label
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.max_concurrent_requests = max_concurrent_requests
        self.model_params = model_params or {}
        self.use_chain_of_thought = use_chain_of_thought
        self.custom_prompt = custom_prompt
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.total_tokens = 0
        self.start_time = None
        self.end_time = None
    
    @abstractmethod
    def create_prompt(self, item: Dict) -> QAPrompt:
        """Create a prompt from the item data"""
        pass
    
    @abstractmethod
    def extract_answer(self, response: str) -> str:
        """Extract the answer from the model's response"""
        pass
    
    @abstractmethod
    def is_correct(self, extracted_answer: str, correct_answer: Any) -> bool:
        """Check if the extracted answer is correct"""
        pass
    
    async def _call_api(self, prompt: str) -> Dict:
        """Call the API and return both the response and token usage"""
        async with self.semaphore:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    **self.model_params
                }
                
                async with session.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    result = await response.json()
                    usage = result.get("usage", {})
                    tokens = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0)
                    }
                    if "choices" not in result:
                        print(result)
                        logger.warning(f"Model {self.label}: No choices found in response")
                        return {
                            "content": "",
                            "tokens": tokens
                        }
                    return {
                        "content": result["choices"][0]["message"]["content"].strip(),
                        "tokens": tokens
                    }
    
    async def evaluate_single(self, item: Dict) -> Dict:
        """Evaluate a single item"""
        prompt = self.create_prompt(item)
        
        try:
            result = await self._call_api(prompt.format_prompt())
            response = result["content"]
            tokens = result["tokens"]
            
            extracted_answer = self.extract_answer(response)
            is_correct = self.is_correct(extracted_answer, item.get("answer_idx", item.get("final_decision")))
            
            r = {
                "question": item.get("question", item.get("QUESTION")),
                "correct_answer": item.get("answer_idx", item.get("final_decision")),
                "model_answer": extracted_answer,
                "full_response": response,
                "is_correct": is_correct,
                "meta_info": item.get("meta_info", ""),
                "tokens": tokens
            }

            if not extracted_answer:
                print(response)
                logger.warning(f"Model {self.label}: No answer found in response")
                r["error"] = "No answer found in response"
            
            return r
        except Exception as e:
            logger.warning(f"Model {self.label}: Evaluation failed - {str(e)}")
            return {
                "question": item.get("question", item.get("QUESTION")),
                "error": str(e),
                "meta_info": item.get("meta_info", ""),
                "is_correct": False
            }
    
    async def evaluate_dataset(self, dataset_path: str, limit: Optional[int] = None) -> List[Dict]:
        """Load and evaluate a dataset"""
        self.start_time = time.time()
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        dataset = []
        
        with open(dataset_path) as f:
            # Try to load as a single JSON object first
            try:
                data = json.load(f)
                # If it's a dictionary with paper IDs as keys (PubMedQA format)
                if isinstance(data, dict) and all(isinstance(k, str) and k.isdigit() for k in data.keys()):
                    dataset = [{"paper_id": k, **v} for k, v in data.items()]
                else:
                    dataset = data
            except json.JSONDecodeError:
                # If that fails, try reading as JSONL
                f.seek(0)  # Reset file pointer
                for i, line in enumerate(f):
                    if limit is not None and i >= limit:
                        break
                    try:
                        item = json.loads(line.strip())
                        if isinstance(item, str):
                            item = json.loads(item)
                        dataset.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line: {e}")
                        continue
        
        if not dataset:
            raise ValueError(f"No valid data found in {dataset_path}")
        
        if limit is not None:
            dataset = dataset[:limit]
        
        tasks = [self.evaluate_single(item) for item in dataset]
        results = []
        
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Evaluating {self.label}"):
            result = await future
            if "tokens" in result:
                self.total_tokens += result["tokens"]["total_tokens"]
                self.total_prompt_tokens += result["tokens"]["prompt_tokens"]
                self.total_completion_tokens += result["tokens"]["completion_tokens"]
            results.append(result)
        
        self.end_time = time.time()
        return results
    
    def generate_report(self, results: List[Dict]) -> pd.DataFrame:
        """Generate a report from the evaluation results"""
        # Calculate accuracy
        correct_answers = sum(1 for r in results if r.get("is_correct", False))
        total_questions = len(results)
        error_count = sum(1 for r in results if "error" in r)
        
        # Calculate token usage and cost
        total_tokens = self.total_tokens
        total_prompt_tokens = self.total_prompt_tokens
        total_completion_tokens = self.total_completion_tokens
        
        # Calculate cost
        total_cost = calculate_token_cost(self.model, total_prompt_tokens, total_completion_tokens)
        cost_per_question = total_cost / total_questions if total_questions > 0 else 0
        
        # Create report
        report = pd.DataFrame([{
            "model": self.model,
            "label": self.label,
            "accuracy": correct_answers / total_questions if total_questions > 0 else 0,
            "correct_answers": correct_answers,
            "total_questions": total_questions,
            "error_count": error_count,
            "use_chain_of_thought": self.use_chain_of_thought,
            "total_tokens": total_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_cost": total_cost,
            "cost_per_question": cost_per_question,
            "evaluation_time": self.end_time - self.start_time if self.end_time else None
        }])
        
        return report 