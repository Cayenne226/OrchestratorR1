import time
from openai import OpenAI


class BaseAgent:
    def __init__(self, model_name: str, cost_per_1m: float, system_prompt: str,
                 api_base: str, api_key: str, timeout: int = 60):
        self.model_name = model_name
        self.cost_per_1m = cost_per_1m
        self.system_prompt = system_prompt
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout
        self._client = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        return self._client

    def call(self, query: str, max_retries: int = 3) -> tuple[str, float]:
        """Call the agent API. Returns (response_text, cost_usd)."""
        client = self._get_client()
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": query},
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                    timeout=self.timeout,
                )
                text = response.choices[0].message.content or ""
                total_tokens = response.usage.total_tokens if response.usage else 0
                cost = total_tokens * self.cost_per_1m / 1_000_000
                return text, cost
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"[Agent error: {str(e)}]", 0.0
                time.sleep(2 ** attempt)
        return "[Agent error: max retries exceeded]", 0.0
