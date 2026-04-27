from langchain.chat_models import init_chat_model
from env_utils import LOCAL_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, OPENAI_API_KEY, OPENAI_BASE_URL, \
    ALIBABA_API_KEY, ALIBABA_BASE_URL

llm = init_chat_model(
    model="deepseek-chat",
    model_provider="openai",
    base_url=DEEPSEEK_BASE_URL,
    api_key=DEEPSEEK_API_KEY,
    temperature=0,
)


# def llm():
#     return None