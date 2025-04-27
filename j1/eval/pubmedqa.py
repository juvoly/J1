from dataclasses import dataclass
from typing import Dict, List, Optional
from .base import QAEvaluator

@dataclass
class PubMedQAPrompt:
    """Prompt for PubMedQA evaluation"""
    question: str
    contexts: List[str]
    use_chain_of_thought: bool = False
    custom_prompt: Optional[str] = None
    
    def format_prompt(self) -> str:
        if self.custom_prompt is not None:
            return self.custom_prompt.format(
                question=self.question,
                contexts=self.contexts
            )
            
        contexts_text = "\n\n".join(self.contexts)
        
        if self.use_chain_of_thought:
            return f"""You are a medical expert. Please answer the following question based on the provided context.

Context:
{contexts_text}

Question: {self.question}

Please follow these steps:
1. First, explain your reasoning step by step
2. Then, provide your final answer in the following format:
Answer: yes/no

For example:
1. The context indicates that X and Y are true
2. Based on this evidence, the answer should be yes
Answer: yes"""
        else:
            return f"""You are a medical expert. Please answer the following question based on the provided context.

Context:
{contexts_text}

Question: {self.question}

Please provide your answer in the following format:
Answer: yes/no

For example:
Answer: yes"""

class PubMedQAEvaluator(QAEvaluator):
    def create_prompt(self, item: Dict) -> PubMedQAPrompt:
        return PubMedQAPrompt(
            question=item["QUESTION"],
            contexts=item["CONTEXTS"],
            use_chain_of_thought=self.use_chain_of_thought,
            custom_prompt=self.custom_prompt
        )
    
    def extract_answer(self, response: str) -> str:
        lines = response.strip().split("\n")
        for line in lines:
            line = line.replace("*", "").lower()
            if line.strip().startswith("answer:"):
                answer = line.strip().split(":")[1].strip()
                return "yes" if answer in ["yes", "y"] else "no"
            if line.strip().startswith("the answer is "):
                answer = line.strip().split("is ")[1].strip().replace(".", "")
                return "yes" if answer in ["yes", "y"] else "no"
        return ""
    
    def is_correct(self, extracted_answer: str, correct_answer: str) -> bool:
        return extracted_answer == correct_answer.lower() 