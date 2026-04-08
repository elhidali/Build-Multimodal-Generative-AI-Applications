import httpx
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# We use ChatOpenAI from langchain_openai to connect to OpenRouter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Initialize the model using OpenRouter credentials and model
llm = ChatOpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model=os.getenv("MODEL_ID", "nvidia/nemotron-3-super-120b-a12b:free"),
    temperature=0.5,
    max_tokens=512,
    http_client=httpx.Client(verify=False)
)


# Define the prompt template
template = """
Generate meeting minutes and a list of tasks based on the provided context.

Context:
{context}

Meeting Minutes:
- Key points discussed
- Decisions made

Task List:
- Actionable items with assignees and deadlines
"""
prompt = ChatPromptTemplate.from_template(template)

# Define the chain
chain = (
    {"context": RunnablePassthrough()}  # Pass the transcript as context
    | prompt
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":
    print(chain.invoke("Hello, how are you?"))