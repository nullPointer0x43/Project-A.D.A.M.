from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from URLs import DEFAULT_OLLAMA_MODEL, DEFAULT_GEMINI_MODEL, DEFAULT_OLLAMA_BASE_URL


class LLM_connector:
    def __init__(self):
        self.llm_ollama = ChatOllama(
            model=DEFAULT_OLLAMA_MODEL,
            temperature=0,
            base_url=DEFAULT_OLLAMA_BASE_URL
        )

        self.llm_gemini = ChatGoogleGenerativeAI(
            model=DEFAULT_GEMINI_MODEL,
            temperature=0
        )