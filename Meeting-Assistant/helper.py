
import httpx
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# We use ChatOpenAI from langchain_openai to connect to OpenRouter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i) < 128)




def product_assistant(ascii_transcript):
    system_prompt = """You are an intelligent assistant specializing in financial products;
    your task is to process transcripts of earnings calls, ensuring that all references to
     financial products and common financial terms are in the correct format. For each
     financial product or common term that is typically abbreviated as an acronym, the full term 
    should be spelled out followed by the acronym in parentheses. For example, '401k' should be
     transformed to '401(k) retirement savings plan', 'HSA' should be transformed to 'Health Savings Account (HSA)' , 'ROA' should be transformed to 'Return on Assets (ROA)', 'VaR' should be transformed to 'Value at Risk (VaR)', and 'PB' should be transformed to 'Price to Book (PB) ratio'. Similarly, transform spoken numbers representing financial products into their numeric representations, followed by the full name of the product in parentheses. For instance, 'five two nine' to '529 (Education Savings Plan)' and 'four zero one k' to '401(k) (Retirement Savings Plan)'. However, be aware that some acronyms can have different meanings based on the context (e.g., 'LTV' can stand for 'Loan to Value' or 'Lifetime Value'). You will need to discern from the context which term is being referred to  and apply the appropriate transformation. In cases where numerical figures or metrics are spelled out but do not represent specific financial products (like 'twenty three percent'), these should be left as is. Your role is to analyze and adjust financial product terminology in the text. Once you've done that, produce the adjusted transcript and a list of the words you've changed"""

    # Concatenate the system prompt and the user transcript
    prompt_input = system_prompt + "\n" + ascii_transcript

    # Create a messages object
    messages = [
        {
            "role": "user",
            "content": prompt_input
        }
    ]

    # Construct the model ID using the environment variable or default
    model_id = os.getenv("MODEL_ID", "meta-llama/llama-3.2-11b-vision-instruct:free")
    
    # Initialize the ChatOpenAI model with OpenRouter details
    llm = ChatOpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=model_id,
        temperature=0.2,
        model_kwargs={"top_p": 0.6},
        http_client=httpx.Client(verify=False)
    )
    
    # Send the input messages to the model and retrieve its response
    response = llm.invoke(messages)
    
    # Extract and return the content of the model's response
    return response.content

