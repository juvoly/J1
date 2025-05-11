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
        
        return f"""{contexts_text}
{self.question}

Choose from:
- yes
- no

When you want to mention the correct answer please do it in this format: 'The correct answer is: <answr>'"""

class PubMedQAEvaluator(QAEvaluator):
    def create_prompt(self, item: Dict) -> PubMedQAPrompt:
        return PubMedQAPrompt(
            question=item["QUESTION"],
            contexts=item["CONTEXTS"],
            use_chain_of_thought=self.use_chain_of_thought,
            custom_prompt=self.custom_prompt
        )
    
    def extract_answer(self, response: str) -> str:
        if "</think>" in response:
            response = response.split("</think>")[1].strip()
        if "answer is" in response:
            for chunk in response.split("answer is"):
                r = chunk.strip().replace(":", "").replace(")", "").replace("*", "").strip().split()[0].lower().replace(".", "")
                if r in ["yes", "no"]:
                    return r
        print(r, response)
        return None
    
    def is_correct(self, extracted_answer: str, correct_answer: str) -> bool:
        return extracted_answer == correct_answer.lower() 