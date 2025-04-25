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
        min_options: int = 2,
        max_options: int = 6,
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
            min_options: Minimum number of options for multiple choice questions (default: 2).
            max_options: Maximum number of options for multiple choice questions (default: 6).
        """
        if not 0 <= mcq_probability <= 1:
            raise ValueError("mcq_probability must be between 0 and 1")
        if min_options < 2:
            raise ValueError("min_options must be at least 2")
        if max_options < min_options:
            raise ValueError("max_options must be greater than or equal to min_options")
        if max_options > 6:
            raise ValueError("max_options cannot exceed 6")
        if openai_kwargs is None:
            openai_kwargs = {}
        if "timeout" not in openai_kwargs:
            openai_kwargs["timeout"] = 300
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
        self.min_options = min_options
        self.max_options = max_options

    def generate_system_prompt(self, row: pd.Series) -> str:
        """
        Generates a system prompt that instructs the model to create either
        a multiple choice question or an open-ended question based on probability.
        Only used if no static system prompt was provided.
        """
        if random.random() < self.mcq_probability:
            num_options = random.randint(self.min_options, self.max_options)
            option_letters = [chr(65 + i) for i in range(num_options)]  # A, B, C, etc.
            option_template = "\n".join([f"{letter}) [option {letter}]" for letter in option_letters])
            
            return f"""You are an expert question generator specializing in creating challenging and thought-provoking multiple choice questions. 
            Generate a multiple choice question with {num_options} options ({', '.join(option_letters)}) based on the provided text. 
            
            Guidelines:
            1. Focus on broader concepts, principles, and implications rather than specific study details
            2. The question should test deep understanding and critical thinking about general medical concepts
            3. Include one correct answer and {num_options-1} plausible but incorrect options
            4. Each option should be well-reasoned and require careful consideration
            5. The correct answer should not be immediately obvious
            6. Consider including options that test common misconceptions
            7. NEVER use specific study details or results - focus on general medical principles
            8. Ensure the question can be understood without needing to see the original text
            9. The question should be about general medical knowledge and principles
            10. Avoid referencing specific studies, numbers, or experimental details
            
            Format your response EXACTLY as follows:
            ```markdown
            Question: [question text]

            {option_template}

            Reasoning:
            [Provide a detailed reasoning dialogue that:
            1. Starts with the key concepts from the question
            2. Discusses why each option is plausible or not
            3. Explains the thought process step by step
            4. Considers different aspects of the problem
            5. Builds a logical argument that leads to the answer
            The reasoning should be a natural dialogue that demonstrates critical thinking]

            Answer: [correct option letter]
            Explanation: [brief explanation of why this is the correct answer]
            ```
            """
        else:
            return """You are an expert question generator specializing in creating thought-provoking open-ended questions. 
            Generate an open-ended question based on the provided text that encourages deep analysis and reasoning about general medical principles.
            
            Guidelines:
            1. Focus on broader concepts, principles, and implications rather than specific study details
            2. The question should require more than just factual recall
            3. Encourage critical thinking about general medical principles
            4. Consider asking about implications, relationships, or patterns in medical practice
            5. The question should be challenging but answerable based on general medical knowledge
            6. NEVER use specific study details or results - focus on general medical principles
            7. Ensure the question can be understood without needing to see the original text
            8. The question should be about general medical knowledge and principles
            9. Avoid referencing specific studies, numbers, or experimental details
            10. Focus on understanding underlying concepts and their clinical implications
            
            Format your response EXACTLY as follows:
            ```markdown
            Question: [question text]

            Reasoning:
            [Provide a detailed reasoning dialogue that:
            1. Starts with the key concepts from the question
            2. Discusses the relevant principles and implications
            3. Explains the thought process step by step
            4. Considers different aspects of the problem
            5. Builds a logical argument that leads to the answer
            The reasoning should be a natural dialogue that demonstrates critical thinking]

            Answer: [detailed answer that includes reasoning and explanation]
            Explanation: [comprehensive explanation using only general medical knowledge]
            ```
            """

    def extract_content(self, response_content: str) -> Dict[str, Any]:
        """
        Extracts the question and answer from the OpenAI response.
        Handles both multiple choice and open-ended question formats.
        Returns a dictionary with empty strings if there's an error in extraction.
        """
        try:
            # Split the response into lines and clean them
            lines = [line.strip() for line in response_content.split('\n') if line.strip()]
            
            question = None
            options = []
            reasoning = None
            answer = None
            explanation = None
            current_section = None
            
            for i, line in enumerate(lines):
                if line.startswith('Question:'):
                    current_section = 'question'
                    question = line.replace('Question:', '').strip()
                    continue
                elif line.startswith('Reasoning:'):
                    current_section = 'reasoning'
                    reasoning = line.replace('Reasoning:', '').strip()
                    continue
                elif line.startswith('Answer:'):
                    current_section = 'answer'
                    answer = line.replace('Answer:', '').strip()
                    continue
                elif line.startswith('Explanation:'):
                    current_section = 'explanation'
                    explanation = line.replace('Explanation:', '').strip()
                    continue
                
                if current_section == 'question':
                    # Check if this line is part of the question or an option
                    if any(line.startswith(f"{chr(65 + i)})") for i in range(26)):
                        options.append(line)
                    elif not question:  # If we haven't found the question yet
                        question = line
                elif current_section == 'reasoning' and not reasoning:
                    reasoning = line
                elif current_section == 'answer' and not answer:
                    answer = line
                elif current_section == 'explanation' and not explanation:
                    explanation = line
            
            # Ensure we have at least a question and answer
            if not question:
                question = ""
            if not answer:
                answer = ""
            
            # Validate answer format for MCQ
            if options:
                # Check if answer is a valid option letter
                answer_letter = answer.strip().upper()
                valid_letters = [chr(65 + i) for i in range(len(options))]
                if answer_letter not in valid_letters:
                    answer = options[0]  # Use first option as fallback
            
            # If we found options, format them into the question
            if options:
                question = f"{question}\n" + "\n".join(options)
            
            # Combine reasoning, answer, and explanation in the correct order
            final_answer = []
            if reasoning:
                final_answer.append(f"Reasoning: {reasoning}")
            final_answer.append(f"Answer: {answer}")
            if explanation:
                final_answer.append(f"Explanation: {explanation}")
            
            result = {
                "question": question,
                "answer": "\n\n".join(final_answer)
            }
            
            return result
        except Exception as e:
            return {"question": "", "answer": ""}

class PatientCaseMapper(PromptMapper):
    """
    A PromptMapper that generates patient case scenarios based on input text.
    Each case includes a patient presentation, relevant history, and key findings.
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
        min_options: int = 2,
        max_options: int = 6,
        mcq_probability: float = 0.5,
    ):
        """
        Initialize the PatientCaseMapper.

        Args:
            input_column: Name of the input column containing the text to generate cases from.
            system_prompt: Optional static system prompt. If None, will generate dynamic prompts.
            openai_model: OpenAI model identifier.
            openai_api_key: OpenAI API key.
            openai_kwargs: Additional kwargs for OpenAI API call.
            max_concurrent_requests: Maximum concurrent OpenAI requests per partition.
            max_tokens: Maximum number of tokens to generate in the response.
            openai_base_url: Custom base URL for OpenAI API.
            min_options: Minimum number of options for multiple choice questions (default: 2).
            max_options: Maximum number of options for multiple choice questions (default: 6).
            mcq_probability: Probability of generating a multiple choice question (default: 0.5).
        """
        if min_options < 2:
            raise ValueError("min_options must be at least 2")
        if max_options < min_options:
            raise ValueError("max_options must be greater than or equal to min_options")
        if max_options > 6:
            raise ValueError("max_options cannot exceed 6")
        if not 0 <= mcq_probability <= 1:
            raise ValueError("mcq_probability must be between 0 and 1")

        if openai_kwargs is None:
            openai_kwargs = {}
        if "timeout" not in openai_kwargs:
            openai_kwargs["timeout"] = 300

        super().__init__(
            input_columns=[input_column],
            output_columns=["question", "answer"],
            user_prompt_template=f"Generate a patient case scenario and related question based on the following text: {{{input_column}}}",
            system_prompt=system_prompt,
            openai_model=openai_model,
            openai_api_key=openai_api_key,
            openai_kwargs=openai_kwargs,
            max_concurrent_requests=max_concurrent_requests,
            max_tokens=max_tokens,
            openai_base_url=openai_base_url,
        )
        self.min_options = min_options
        self.max_options = max_options
        self.mcq_probability = mcq_probability

    def generate_system_prompt(self, row: pd.Series) -> str:
        """
        Generates a system prompt that instructs the model to create a patient case scenario
        and a related question (either multiple-choice or open-ended based on probability).
        """
        is_mcq = random.random() < self.mcq_probability
        
        if is_mcq:
            num_options = random.randint(self.min_options, self.max_options)
            option_letters = [chr(65 + i) for i in range(num_options)]  # A, B, C, etc.
            option_template = "\n".join([f"{letter}) [option {letter}]" for letter in option_letters])
            
            return f"""You are an expert medical educator specializing in creating realistic patient case scenarios and exam-style questions.
            Generate a detailed patient case followed by a challenging multiple-choice question with {num_options} options.

            Guidelines:
            1. Create a realistic patient presentation that could be encountered in clinical practice
            2. Include relevant patient information naturally in the narrative
            3. Focus on broader clinical concepts rather than specific study details
            4. Make the case challenging but realistic
            5. Avoid using specific study results or numbers
            6. Ensure the case is educational and promotes critical thinking
            7. The case should be understandable without needing to see the original text
            8. Focus on clinical reasoning and decision-making
            9. Include realistic patient responses and behaviors
            10. The question should test clinical reasoning and decision-making

            Patient Case Format:
            Vary your cases significantly in terms of structure and detail level. Some cases should be:
            - Very detailed with comprehensive history and exam findings
            - Brief and focused on key presenting symptoms
            - Medium length with selective important details
            - Focused on a specific aspect of care
            - Emphasizing different aspects (history vs exam vs lab results)
            
            Vary the inclusion of:
            - Patient demographics (sometimes include age/gender, sometimes not)
            - Medical history (sometimes detailed, sometimes minimal)
            - Physical exam findings (sometimes comprehensive, sometimes focused)
            - Laboratory results (sometimes included, sometimes omitted)
            - Social history (sometimes detailed, sometimes brief)
            - Family history (sometimes included, sometimes omitted)
            
            Vary the presentation style:
            - Some cases should be narrative and story-like
            - Some should be more clinical and structured
            - Some should focus on a specific clinical encounter
            - Some should span multiple visits
            - Some should be emergency presentations
            - Some should be routine follow-ups
            
            Vary the clinical setting:
            - Emergency department
            - Primary care clinic
            - Specialty consultation
            - Hospital ward
            - Community health center
            - Telemedicine visit
            
            Format your response EXACTLY as follows:
            ```markdown
            Patient Case:
            [Write a natural narrative describing the patient's situation, including relevant history and findings]

            Question:
            [A challenging multiple-choice question based on the case]

            {option_template}

            Reasoning:
            [Provide a detailed reasoning dialogue that:
            1. Starts with the key findings from the case
            2. Discusses the differential diagnosis
            3. Explains why each option is plausible or not
            4. Identifies the most likely diagnosis
            5. Justifies the final answer with clear clinical reasoning
            The reasoning should be a natural dialogue that leads directly to the answer]

            Answer: [correct option letter]
            Explanation: [brief explanation of why this is the correct answer]
            ```
            """
        else:
            return """You are an expert medical educator specializing in creating realistic patient case scenarios and exam-style questions.
            Generate a detailed patient case followed by a challenging open-ended question.

            Guidelines:
            1. Create a realistic patient presentation that could be encountered in clinical practice
            2. Include relevant patient information naturally in the narrative
            3. Focus on broader clinical concepts rather than specific study details
            4. Make the case challenging but realistic
            5. Avoid using specific study results or numbers
            6. Ensure the case is educational and promotes critical thinking
            7. The case should be understandable without needing to see the original text
            8. Focus on clinical reasoning and decision-making
            9. Include realistic patient responses and behaviors
            10. The question should require critical thinking and analysis

            Patient Case Format:
            Vary your cases significantly in terms of structure and detail level. Some cases should be:
            - Very detailed with comprehensive history and exam findings
            - Brief and focused on key presenting symptoms
            - Medium length with selective important details
            - Focused on a specific aspect of care
            - Emphasizing different aspects (history vs exam vs lab results)
            
            Vary the inclusion of:
            - Patient demographics (sometimes include age/gender, sometimes not)
            - Medical history (sometimes detailed, sometimes minimal)
            - Physical exam findings (sometimes comprehensive, sometimes focused)
            - Laboratory results (sometimes included, sometimes omitted)
            - Social history (sometimes detailed, sometimes brief)
            - Family history (sometimes included, sometimes omitted)
            
            Vary the presentation style:
            - Some cases should be narrative and story-like
            - Some should be more clinical and structured
            - Some should focus on a specific clinical encounter
            - Some should span multiple visits
            - Some should be emergency presentations
            - Some should be routine follow-ups
            
            Vary the clinical setting:
            - Emergency department
            - Primary care clinic
            - Specialty consultation
            - Hospital ward
            - Community health center
            - Telemedicine visit
            
            Format your response EXACTLY as follows:
            ```markdown
            Patient Case:
            [Write a natural narrative describing the patient's situation, including relevant history and findings]

            Question:
            [A challenging open-ended question based on the case]

            Reasoning:
            [Provide a detailed reasoning dialogue that:
            1. Starts with the key findings from the case
            2. Discusses the relevant clinical concepts
            3. Explains the thought process step by step
            4. Considers different aspects of the problem
            5. Builds a logical argument that leads to the answer
            The reasoning should be a natural dialogue that demonstrates clinical thinking]

            Answer: [detailed answer that includes reasoning and explanation]
            Explanation: [comprehensive explanation using only general medical knowledge]
            ```
            """

    def extract_content(self, response_content: str) -> Dict[str, Any]:
        """
        Extracts the patient case and answer components from the OpenAI response.
        Returns a dictionary with empty strings if there's an error in extraction.
        """
        try:
            # Split the response into lines and clean them
            lines = [line.strip() for line in response_content.split('\n') if line.strip()]
            
            case_parts = []
            question = None
            options = []
            reasoning = None
            answer = None
            explanation = None
            current_section = None
            
            for i, line in enumerate(lines):
                if line.startswith('Patient Case:'):
                    current_section = 'case'
                    continue
                elif line.startswith('Question:'):
                    current_section = 'question'
                    question = line.replace('Question:', '').strip()
                    continue
                elif line.startswith('Reasoning:'):
                    current_section = 'reasoning'
                    reasoning = line.replace('Reasoning:', '').strip()
                    continue
                elif line.startswith('Answer:'):
                    current_section = 'answer'
                    answer = line.replace('Answer:', '').strip()
                    continue
                elif line.startswith('Explanation:'):
                    current_section = 'explanation'
                    explanation = line.replace('Explanation:', '').strip()
                    continue
                
                if current_section == 'case':
                    case_parts.append(line)
                elif current_section == 'question':
                    # Check if this line is part of the question or an option
                    if any(line.startswith(f"{chr(65 + i)})") for i in range(26)):
                        options.append(line)
                    elif not question:  # If we haven't found the question yet
                        question = line
                elif current_section == 'reasoning' and not reasoning:
                    reasoning = line
                elif current_section == 'answer' and not answer:
                    answer = line
                elif current_section == 'explanation' and not explanation:
                    explanation = line
            
            # Format the question with options if they exist
            formatted_question = question or ""
            if options:
                formatted_question = f"{formatted_question}\n" + "\n".join(options)
            
            # Combine all case details into a single question
            final_question = "Patient Case:\n\n" + "\n".join(case_parts) + "\n\nQuestion:\n" + formatted_question
            
            # Combine reasoning, answer, and explanation into a single answer
            final_answer = []
            if reasoning:
                final_answer.append(f"Reasoning: {reasoning}")
            final_answer.append(f"Answer: {answer or ''}")
            if explanation:
                final_answer.append(f"Explanation: {explanation}")
            
            # Ensure we have valid content
            if not final_question.strip() or not final_answer:
                return {"question": "", "answer": ""}
            
            result = {
                "question": final_question,
                "answer": "\n\n".join(final_answer)
            }
            
            return result
        except Exception as e:
            return {"question": "", "answer": ""} 