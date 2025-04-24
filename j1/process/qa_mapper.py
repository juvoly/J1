import random
from typing import Dict, Any, List, Optional

import pandas as pd
from j1.process.mappers import PromptMapper

class QAPromptMapper(PromptMapper):
    """
    A PromptMapper that generates questions and answers based on input text.
    With a 50% probability, it will generate either a multiple choice question
    or an open-ended question.
    """

    def __init__(
        self,
        input_column: str,
        system_prompt: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
        openai_api_key: str = None,
        openai_kwargs: Dict[str, Any] = None,
        max_concurrent_requests: int = 10,
        max_tokens: int = 1000,
        openai_base_url: str = None,
        mcq_probability: float = 0.5,
    ):
        """
        Initialize the QAPromptMapper.

        Args:
            input_columns: List of input column names containing the text to generate questions from.
            system_prompt: Optional static system prompt. If None, will generate dynamic prompts.
            openai_model: OpenAI model identifier.
            openai_api_key: OpenAI API key.
            openai_kwargs: Additional kwargs for OpenAI API call.
            max_concurrent_requests: Maximum concurrent OpenAI requests per partition.
            max_tokens: Maximum number of tokens to generate in the response.
            openai_base_url: Custom base URL for OpenAI API.
            mcq_probability: Probability of generating a multiple choice question (default: 0.5).
        """
        if not 0 <= mcq_probability <= 1:
            raise ValueError("mcq_probability must be between 0 and 1")

        super().__init__(
            input_columns=[input_column],
            output_columns=["question", "answer"],
            user_prompt_template=f"Generate a question and answer based on the following text: {{{input_column}}}",
            system_prompt=system_prompt,
            openai_model=openai_model,
            openai_api_key=openai_api_key,
            openai_kwargs=openai_kwargs,
            max_concurrent_requests=max_concurrent_requests,
            max_tokens=max_tokens,
            openai_base_url=openai_base_url,
        )
        self.mcq_probability = mcq_probability

    def generate_system_prompt(self, row: pd.Series) -> str:
        """
        Generates a system prompt that instructs the model to create either
        a multiple choice question or an open-ended question based on probability.
        Only used if no static system prompt was provided.
        """
        if random.random() < self.mcq_probability:
            return """You are an expert question generator specializing in creating challenging and thought-provoking multiple choice questions. 
            Generate a multiple choice question with 4 options (A, B, C, D) based on the provided text. 
            
            Guidelines:
            1. The question should test deep understanding and critical thinking
            2. Include one correct answer and three plausible but incorrect options
            3. Each option should be well-reasoned and require careful consideration
            4. The correct answer should not be immediately obvious
            5. Consider including options that test common misconceptions
            6. NEVER use ambiguous references like "this study" or "the research" - always include specific context
            7. If referencing a study or source, include its key details in the question
            8. Ensure the question can be understood without needing to see the original text
            9. Provide context ONLY when necessary:
               - Include context for specific experimental setups, novel techniques, or unique study conditions
               - Omit context for general medical knowledge, common procedures, or well-established concepts
               - Use your medical expertise to determine if the knowledge is general or specific
            10. The context should include any relevant study details, experimental setup, or background information needed to understand the question
            
            Format your response EXACTLY as follows:
            ```markdown
            Context: [Provide any necessary background information, study details, or experimental setup needed to understand the question. If the question tests general medical knowledge, write "None"]

            Question: [question text]

            A) [option A]
            B) [option B]
            C) [option C]
            D) [option D]

            Answer: [correct option letter]
            Explanation: [brief explanation of why this is the correct answer, using only information provided in the context or general medical knowledge]
            ```
            
            Important:
            - Do not include explanations within the options
            - Keep the options concise and clear
            - The explanation should be separate from the options
            - Follow the markdown format exactly
            - Only provide context when necessary for understanding specific experimental details
            - For general medical knowledge questions, omit context to encourage memorization of core concepts
            - In explanations, only reference specific numbers or details that are explicitly provided in the context
            - When specific numbers are not provided, use general terms like "significant", "moderate", or "minimal" instead"""
        else:
            return """You are an expert question generator specializing in creating thought-provoking open-ended questions. 
            Generate an open-ended question based on the provided text that encourages deep analysis and reasoning.
            
            Guidelines:
            1. The question should require more than just factual recall
            2. Encourage critical thinking and analysis
            3. The answer should demonstrate understanding of underlying concepts
            4. Consider asking about implications, relationships, or patterns
            5. The question should be challenging but answerable based on the text
            6. NEVER use ambiguous references like "this study" or "the research" - always include specific context
            7. If referencing a study or source, include its key details in the question
            8. Ensure the question can be understood without needing to see the original text
            9. Provide context ONLY when necessary:
               - Include context for specific experimental setups, novel techniques, or unique study conditions
               - Omit context for general medical knowledge, common procedures, or well-established concepts
               - Use your medical expertise to determine if the knowledge is general or specific
            10. The context should include any relevant study details, experimental setup, or background information needed to understand the question
            
            Format your response EXACTLY as follows:
            ```markdown
            Context: [Provide any necessary background information, study details, or experimental setup needed to understand the question. If the question tests general medical knowledge, write "None"]

            Question: [question text]

            Answer: [detailed answer that includes reasoning and explanation, using only information provided in the context or general medical knowledge]
            ```
            
            Important:
            - Keep the question and answer clearly separated
            - The answer should be comprehensive and well-reasoned
            - Follow the markdown format exactly
            - Only provide context when necessary for understanding specific experimental details
            - For general medical knowledge questions, omit context to encourage memorization of core concepts
            - In answers, only reference specific numbers or details that are explicitly provided in the context
            - When specific numbers are not provided, use general terms like "significant", "moderate", or "minimal" instead
            - Focus on general principles and implications rather than specific numerical findings
            - If the context doesn't provide specific numbers, discuss trends and patterns in general terms"""

    def extract_content(self, response_content: str) -> Dict[str, Any]:
        """
        Extracts the question and answer from the OpenAI response.
        Handles both multiple choice and open-ended question formats.
        """
        try:
            # Split the response into lines and clean them
            lines = [line.strip() for line in response_content.split('\n') if line.strip()]
            
            # Find the context, question, options (if MCQ), and answer lines
            context = None
            question = None
            options = []
            answer = None
            
            for i, line in enumerate(lines):
                if line.startswith('Context:'):
                    context = line.replace('Context:', '').strip()
                elif line.startswith('Question:'):
                    question = line.replace('Question:', '').strip()
                elif line.startswith(('A)', 'B)', 'C)', 'D)')) and not any(x in line.lower() for x in ['correct', 'incorrect', 'because', 'this is']):
                    # Only include option lines that don't contain explanations
                    options.append(line)
                elif line.startswith('Answer:'):
                    answer = line.replace('Answer:', '').strip()
            
            if not question or not answer:
                raise ValueError("Could not find question or answer in response")
            
            # If we found options, format them into the question
            if options:
                question = f"{question}\n" + "\n".join(options)
            
            # If context is provided, prepend it to the question
            if context and context.lower() != 'none':
                question = f"{context}\n\n{question}"
            
            return {
                "question": question,
                "answer": answer
            }
        except Exception as e:
            return {
                "question": f"Error extracting question: {str(e)}",
                "answer": f"Error extracting answer: {str(e)}"
            } 