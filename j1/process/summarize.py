from typing import Any, Dict, List, Optional

from .mappers import PromptMapper


class SummaryMapper(PromptMapper):
    """
    A specialized PromptMapper that generates a concise summary of text
    from an input column.
    """
    DEFAULT_SYSTEM_PROMPT = "You are an expert knowledge gatherer. Distill the key information from the provided text. Respond ONLY with 'Knowledge: ' followed by a paraphrased version of the text. Avoid mentioning 'the text starts with' or 'the study finds' - write your knowledge more from a factual perspective, as if you are a librarian. Avoid using words like 'the' or 'this' if it is not obvious what it is referring to."
    DEFAULT_USER_PROMPT_TEMPLATE = "Please distill knowledge from the following text:\\n\\n{text_content}"

    def __init__(
        self,
        input_column: str,
        output_column: str = "summary",
        openai_model: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
        openai_kwargs: Optional[Dict[str, Any]] = None,
        max_concurrent_requests: int = 10,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE,
        max_tokens: int = 1600,
        openai_base_url: Optional[str] = None,
    ):
        """Initialize the SummaryMapper.

        Args:
            input_column: Name of the column containing the text to summarize.
                          This will be used to format the user_prompt_template.
            output_column: Name of the new column for the generated summary.
            openai_model: OpenAI model identifier.
            openai_api_key: OpenAI API key (or reads OPENAI_API_KEY env var).
            openai_kwargs: Additional kwargs for OpenAI API call.
            max_concurrent_requests: Maximum concurrent OpenAI requests per partition.
            system_prompt: System prompt content. Defaults to a summarization prompt.
            user_prompt_template: f-string template for user prompt. Must contain
                                  a placeholder matching the input_column name
                                  (e.g., {text_content} if input_column is 'text_content').
                                  Defaults to a summarization template.
            max_tokens: Maximum number of tokens to generate in the response.
            openai_base_url: Custom base URL for OpenAI API (e.g., for local servers).
        """
        # Ensure the default template uses the provided input_column name
        if user_prompt_template == self.DEFAULT_USER_PROMPT_TEMPLATE:
             user_prompt_template = f"Please distill knowledge from the following text:\\n\\n{{{input_column}}}"

        super().__init__(
            input_columns=[input_column],  # SummaryMapper uses a single input column
            output_columns=[output_column],
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            openai_model=openai_model,
            openai_api_key=openai_api_key,
            openai_kwargs=openai_kwargs,
            max_concurrent_requests=max_concurrent_requests,
            max_tokens=max_tokens,
            openai_base_url=openai_base_url,
        ) 

    def extract_content(self, response_content: str) -> Dict[str, Any]:
        """
        Extracts the summary text from the OpenAI response.
        Assumes the response starts with "Knowledge: ".
        """
        prefix = "Knowledge: "
        if response_content.startswith(prefix):
            summary = response_content[len(prefix):].strip()
        else:
            # If the prefix is missing, return the whole content as a fallback
            summary = response_content.strip()
        return {self.output_columns[0]: summary} 