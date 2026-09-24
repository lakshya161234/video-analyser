import os


def get_llm(temperature: float = 0.2):
    provider = os.getenv("LLM_PROVIDER", "mistral").strip().lower()

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set GOOGLE_API_KEY or GEMINI_API_KEY in .env to use Gemini."
            )

        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.7-flash"),
            api_key=api_key,
            temperature=temperature,
        )

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Set GROQ_API_KEY in .env to use Groq.")

        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            api_key=api_key,
            temperature=temperature,
        )

    if provider != "mistral":
        raise ValueError("LLM_PROVIDER must be 'mistral', 'gemini', or 'groq'.")

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set in environment / .env")

    from langchain_mistralai import ChatMistralAI

    return ChatMistralAI(
        model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        mistral_api_key=api_key,
        temperature=temperature,
    )
