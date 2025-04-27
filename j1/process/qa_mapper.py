import random
import logging
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from j1.process.base import Processor

logger = logging.getLogger(__name__)

class PatientMapper(Processor):
    """
    A Processor that generates patient case scenarios in three steps:
    1. Generate a patient case with demographics
    2. Generate a multiple choice question based on the case
    3. Generate chain-of-thought reasoning for the answer
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
        n_repetitions: int = 1,
        samples_per_partition: int = 10,
    ):
        """
        Initialize the PatientMapper.

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
            n_repetitions: Number of different cases to generate per input row (default: 1).
            samples_per_partition: Number of samples to take from each partition (default: 10).
        """
        if min_options < 2:
            raise ValueError("min_options must be at least 2")
        if max_options < min_options:
            raise ValueError("max_options must be greater than or equal to min_options")
        if max_options > 6:
            raise ValueError("max_options cannot exceed 6")
        if n_repetitions < 1:
            raise ValueError("n_repetitions must be at least 1")
        if samples_per_partition < 1:
            raise ValueError("samples_per_partition must be at least 1")

        if openai_kwargs is None:
            openai_kwargs = {}
        if "timeout" not in openai_kwargs:
            openai_kwargs["timeout"] = 300

        self.input_column = input_column
        self.system_prompt = system_prompt
        self.openai_model = openai_model
        self.openai_api_key = openai_api_key
        self.openai_kwargs = openai_kwargs
        self.max_concurrent_requests = max_concurrent_requests
        self.max_tokens = max_tokens
        self.openai_base_url = openai_base_url
        self.min_options = min_options
        self.max_options = max_options
        self.n_repetitions = n_repetitions
        self.samples_per_partition = samples_per_partition

    def _generate_demographics(self) -> Dict[str, Any]:
        """Generate random patient demographics."""
        # Generate age with different ranges and probabilities
        age_ranges = [
            (0.0, 1.0, 0.1),  # Infants (0-1 year, in months)
            (1.0, 5.0, 0.1),  # Toddlers (1-5 years, in years)
            (5.0, 18.0, 0.1),  # Children (5-18 years, in years)
            (18.0, 90.0, 1.0),  # Adults (18-90 years, in years)
        ]
        
        # Weight the probability of each age range
        weights = [0.15, 0.15, 0.2, 0.5]  # Higher weight for adults
        selected_range = random.choices(age_ranges, weights=weights)[0]
        
        # Generate age within the selected range
        min_age, max_age, step = selected_range
        age = round(random.uniform(min_age, max_age), 1)
        
        # Format age string based on the range
        if age < 2.0:
            age_str = f"{int(age * 12)} months"
        else:
            age_str = f"{age} years"
        
        # Generate gender
        gender = random.choice(["male", "female"])
        
        return {
            "age": age,
            "age_str": age_str,
            "gender": gender,
            "pronoun": "he" if gender == "male" else "she",
            "possessive": "his" if gender == "male" else "her"
        }

    def generate_system_prompt(self, row: pd.Series) -> str:
        """
        Generates a system prompt that instructs the model to create a patient case scenario.
        This is the first step in the three-step generation process.
        """
        demographics = self._generate_demographics()
        input_text = row[self.input_column]
        
        return f"""You are an expert medical educator specializing in creating realistic patient case scenarios.
        Generate a detailed patient case based on the following medical concept or topic:

        {input_text}

        Patient Demographics:
        - Age: {demographics['age_str']}
        - Gender: {demographics['gender']}
        - Pronoun: {demographics['pronoun']}
        - Possessive: {demographics['possessive']}

        Guidelines:
        1. Create a realistic patient presentation that demonstrates the medical concept from the input text
        2. Include relevant patient information naturally in the narrative
        3. Focus on clinical aspects directly related to the input text
        4. Make the case challenging but realistic
        5. Ensure the case is educational and promotes critical thinking
        6. The case should be understandable without needing to see the original text
        7. Focus on clinical reasoning and decision-making
        8. Include realistic patient responses and behaviors
        9. For pediatric cases, include appropriate developmental context and caregiver interactions

        Format your response EXACTLY as follows:
        <case>
        [Write a natural narrative describing the patient's situation, including:
        - Presenting complaint
        - Relevant history
        - Key findings
        - Any other relevant details]
        </case>
        """

    def generate_mcq_prompt(self, patient_case: str, input_text: str) -> str:
        """
        Generates a prompt for creating a multiple choice question based on the patient case.
        This is the second step in the three-step generation process.
        """
        num_options = random.randint(self.min_options, self.max_options)
        option_letters = [chr(65 + i) for i in range(num_options)]  # A, B, C, etc.
        option_template = "\n".join([f"{letter}) [option {letter}]" for letter in option_letters])
        
        return f"""Based on the following patient case and medical concept, generate a challenging multiple-choice question with {num_options} options.

        Medical Concept:
        {input_text}

        Patient Case:
        {patient_case}

        Guidelines:
        1. The question should test clinical reasoning and decision-making related to the medical concept
        2. Include one correct answer and {num_options-1} plausible but incorrect options
        3. Each option should be well-reasoned and require careful consideration
        4. The correct answer should not be immediately obvious
        5. Consider including options that test common misconceptions about the medical concept
        6. The question should be challenging but answerable based on the case
            
        Format your response EXACTLY as follows:
        <question>
            [A challenging multiple-choice question based on the case]

            {option_template}
        </question>
        """

    def generate_cot_prompt(self, patient_case: str, question: str, options: List[str], input_text: str) -> str:
        """
        Generates a prompt for creating chain-of-thought reasoning for the question.
        This is the third step in the three-step generation process.
        """
        return f"""Based on the following patient case, question, and medical concept, provide a detailed chain-of-thought reasoning
        that leads to the correct answer. Structure your reasoning using XML elements for each step.

        Medical Concept:
        {input_text}

        Patient Case:
        {patient_case}

        Question:
        {question}

        Options:
        {chr(10).join(options)}

        Guidelines:
        1. Start with the key findings from the case
        2. Discuss the differential diagnosis
        3. Explain why each option is plausible or not
        4. Identify the most likely diagnosis
        5. Justify the final answer with clear clinical reasoning
        6. The reasoning should be a natural dialogue that leads directly to the answer
        7. Use XML elements to structure each step of your reasoning
        8. Ensure the reasoning directly relates to the medical concept from the input text
            
        Format your response EXACTLY as follows:
        <thinking>
        <key_findings>
        [List and analyze the key findings from the case]
        </key_findings>

        <differential_diagnosis>
        [List and discuss potential diagnoses based on the findings]
        </differential_diagnosis>

        <option_analysis>
        [Analyze each option in detail:
        A) [Analysis of option A]
        B) [Analysis of option B]
        C) [Analysis of option C]
        ...]
        </option_analysis>

        <clinical_reasoning>
        [Provide step-by-step clinical reasoning that:
        1. Connects the findings to the diagnosis
        2. Explains why other options are less likely
        3. Justifies the final answer]
        </clinical_reasoning>

        <conclusion>
        [Summarize the reasoning and state the final answer]
        </conclusion>
        </thinking>

        <answer>
        [correct option letter]
        </answer>
        """

    def _extract_case(self, response_content: str) -> str:
        """Extract the patient case from the response."""
        try:
            lines = [line.strip() for line in response_content.split('\n') if line.strip()]
            case = []
            in_case = False
            
            for line in lines:
                if line.startswith('<case>'):
                    in_case = True
                    continue
                elif line.startswith('</case>'):
                    break
                elif in_case:
                    case.append(line)
            
            return "\n".join(case) if case else ""
        except Exception:
            return ""

    def _extract_question_and_options(self, response_content: str) -> Tuple[str, List[str]]:
        """Extract the question and options from the response."""
        try:
            lines = [line.strip() for line in response_content.split('\n') if line.strip()]
            question = []
            options = []
            in_question = False
            
            for line in lines:
                if line.startswith('<question>'):
                    in_question = True
                    continue
                elif line.startswith('</question>'):
                    break
                elif in_question:
                    if any(line.startswith(f"{chr(65 + i)})") for i in range(26)):
                        options.append(line)
                    else:
                        question.append(line)
            
            return "\n".join(question), options
        except Exception:
            return "", []

    def _extract_reasoning_and_answer(self, response_content: str) -> Tuple[str, str]:
        """Extract the reasoning and answer from the response."""
        try:
            # Split the content into thinking and answer sections
            thinking_section = ""
            answer = ""
            
            # Find the thinking section
            thinking_start = response_content.find("<thinking>")
            thinking_end = response_content.find("</thinking>")
            if thinking_start != -1 and thinking_end != -1:
                thinking_section = response_content[thinking_start:thinking_end + len("</thinking>")]
            
            # Find the answer section
            answer_start = response_content.find("<answer>")
            answer_end = response_content.find("</answer>")
            if answer_start != -1 and answer_end != -1:
                answer = response_content[answer_start + len("<answer>"):answer_end].strip()
            
            if not thinking_section or not answer:
                logger.error(f"Missing sections in response. Thinking section: {bool(thinking_section)}, Answer: {answer}")
                return "", ""
            
            return thinking_section, answer
        except Exception as e:
            logger.error(f"Error extracting reasoning and answer: {e}")
            logger.error(f"Response content: {response_content[:500]}...")  # Log first 500 chars of response
            return "", ""

    def _get_randomized_kwargs(self) -> Dict[str, Any]:
        """Get OpenAI kwargs with randomized temperature."""
        kwargs = self.openai_kwargs.copy()
        kwargs["temperature"] = random.uniform(0.4, 0.5)
        return kwargs

    def _log_generation_failure(self, step: str, error: Exception, input_text: str) -> None:
        """Log a warning when generation fails."""
        logger.warning(
            f"Failed to generate {step} for input: {input_text[:100]}... Error: {str(error)}"
        )

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the input DataFrame to generate patient cases, questions, reasoning, and answers.
        Uses map_partitions to handle Dask DataFrames and limits processing to samples_per_partition samples per partition.
        """
        import asyncio
        from openai import AsyncOpenAI
        from typing import List

        async def process_row(client: AsyncOpenAI, row: pd.Series) -> List[Dict[str, str]]:
            results = []
            for rep_idx in range(self.n_repetitions):
                try:
                    input_text = row[self.input_column]
                    kwargs = self._get_randomized_kwargs()
                    
                    # Step 1: Generate patient case
                    prompt = self.generate_system_prompt(row)
                    case_response = await client.chat.completions.create(
                        model=self.openai_model,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                        **kwargs,
                    )
                    case = self._extract_case(case_response.choices[0].message.content)
                    if not case:
                        raise ValueError("Failed to extract patient case")

                    # Step 2: Generate MCQ
                    mcq_prompt = self.generate_mcq_prompt(case, input_text)
                    mcq_response = await client.chat.completions.create(
                        model=self.openai_model,
                        messages=[
                            {"role": "user", "content": mcq_prompt},
                        ],
                        **kwargs,
                    )
                    question, options = self._extract_question_and_options(mcq_response.choices[0].message.content)
                    if not question or not options:
                        raise ValueError("Failed to extract question and options")

                    # Step 3: Generate reasoning and answer
                    cot_prompt = self.generate_cot_prompt(case, question, options, input_text)
                    cot_response = await client.chat.completions.create(
                        model=self.openai_model,
                        messages=[
                            {"role": "user", "content": cot_prompt},
                        ],
                        **kwargs,
                    )
                    reasoning, answer = self._extract_reasoning_and_answer(cot_response.choices[0].message.content)
                    if not reasoning or not answer:
                        print(cot_response.choices[0].message.content)
                        raise ValueError("Failed to extract reasoning and answer")

                    results.append({
                        "case": case,
                        "question": f"{question}\n" + "\n".join(options),
                        "reasoning": reasoning,
                        "answer": answer
                    })
                except Exception as e:
                    self._log_generation_failure(f"repetition {rep_idx + 1}", e, input_text)
                    results.append(None)
            return results

        async def process_partition(partition: pd.DataFrame) -> pd.DataFrame:
            # Sample the partition if it's larger than samples_per_partition
            if len(partition) > self.samples_per_partition:
                partition = partition.sample(n=self.samples_per_partition, random_state=42)
                logger.info(f"Sampled {self.samples_per_partition} rows from partition for processing")

            client = AsyncOpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            
            async def process_with_semaphore(row: pd.Series):
                async with semaphore:
                    return await process_row(client, row)
            
            tasks = [process_with_semaphore(row) for _, row in partition.iterrows()]
            results = await asyncio.gather(*tasks)
            
            # Create output DataFrame for this partition
            flattened_results = []
            for row_idx, row_results in enumerate(results):
                for rep_idx, result in enumerate(row_results):
                    if result is None:
                        continue
                    flattened_results.append({
                        **result,
                        "repetition_index": str(rep_idx)  # Convert to string
                    })
            
            if not flattened_results:
                return pd.DataFrame(columns=["filename", "case", "question", "reasoning", "answer", "repetition_index"])
            
            # Create output DataFrame with new indices
            output_df = pd.DataFrame(flattened_results)
            
            # Create a new DataFrame with just the filename column
            filename_df = partition[["filename"]].reset_index(drop=True)
            
            # Create a new DataFrame with the same number of rows as output_df
            # by repeating the filename rows for each repetition
            repeated_filenames = pd.DataFrame({
                "filename": filename_df["filename"].repeat(self.n_repetitions).reset_index(drop=True)
            })
            
            # Combine the repeated filenames with the output data
            result_df = pd.concat([repeated_filenames, output_df], axis=1)
            
            # Ensure all columns are of string type
            for col in ["filename", "case", "question", "reasoning", "answer", "repetition_index"]:
                if col in result_df.columns:
                    result_df[col] = result_df[col].astype(str)
            
            return result_df

        def sync_process_partition(partition: pd.DataFrame) -> pd.DataFrame:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(process_partition(partition))
            finally:
                loop.close()

        # Define metadata for the output DataFrame
        meta = pd.DataFrame(columns=["filename", "case", "question", "reasoning", "answer", "repetition_index"])
        
        # Apply map_partitions
        return df.map_partitions(
            sync_process_partition,
            meta=meta,
            enforce_metadata=True
        ) 