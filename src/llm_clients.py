import re
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types

class BaseLLMProvider:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.config = kwargs

    def generate(self, prompt, max_tokens=1000, temperature=0.7):
        raise NotImplementedError("Subclass must implement generate()")

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        self.client = OpenAI(
            api_key=kwargs.get('api_key'),
            base_url=kwargs.get('base_url')
        )

    def generate(self, prompt, max_tokens=1000, temperature=0.7):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        text = response.choices[0].message.content
        if not text:
            return '[]'
        return re.sub(r'```json\s?|```', '', text).strip()

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        self.client = Anthropic(api_key=kwargs.get('api_key'))

    def generate(self, prompt, max_tokens=1000, temperature=0.7):
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=temperature
        )
        text = response.content[0].text
        if not text:
            return '[]'
        return re.sub(r'```json\s?|```', '', text).strip()

class GoogleProvider(BaseLLMProvider):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        self.client = genai.Client(api_key=kwargs.get('api_key'))

    def generate(self, prompt, max_tokens=8192, temperature=0.1):
        effective_tokens = max(max_tokens, 8192)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=effective_tokens,
                temperature=temperature,
                response_mime_type='application/json',
                system_instruction="Eres un exportador de datos JSON puro. No expliques, no razones, no pienses en voz alta. Genera directamente el JSON solicitado."
            )
        )
        text = response.text
        if not text:
            return '[]'
        clean_text = re.sub(r'```json\s?|```', '', text).strip()
        return clean_text


PROVIDER_REGISTRY = {
    'openai': OpenAIProvider,
    'anthropic': AnthropicProvider,
    'google': GoogleProvider,
}

def create_llm_provider(provider_type, model_name, **kwargs):
    if provider_type not in PROVIDER_REGISTRY:
        raise ValueError(f"Proveedor desconocido: {provider_type}")
    return PROVIDER_REGISTRY[provider_type](model_name, **kwargs)
