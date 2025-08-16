from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from .config import settings

llm = ChatGoogleGenerativeAI(
    model="gemini-1.0-pro",
    temperature=0, 
    google_api_key=settings.GOOGLE_API_KEY
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=settings.GOOGLE_API_KEY
)