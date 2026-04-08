import httpx
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# We use ChatOpenAI from langchain_openai to connect to OpenRouter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Initialize the model using OpenRouter credentials and model
llm = ChatOpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model=os.getenv("MODEL_ID", "nvidia/nemotron-3-super-120b-a12b:free"),
    temperature=0.5,
    max_tokens=512,
    http_client=httpx.Client(verify=False)
)

# Define a prompt template (adapted from langchain.prompts import PromptTemplate)
prompt = PromptTemplate.from_template("How to read a {topic} effectively?")

# Create a chain using LangChain Expression Language (LCEL) instead of LLMChain
chain = prompt | llm

# Invoke the chain
response = chain.invoke({"topic": "book"})
print(response.content)
