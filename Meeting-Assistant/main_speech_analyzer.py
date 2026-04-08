import torch
import os
import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline  # For Speech-to-Text
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

#######------------- LLM Initialization-------------#######

llm = ChatOpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model=os.getenv("MODEL_ID", "nvidia/nemotron-3-super-120b-a12b:free"),
    temperature=0.5,
    max_tokens=2048,
    http_client=httpx.Client(verify=False)
)

#######------------- Helper Functions-------------#######

# Function to remove non-ASCII characters
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
    model_id = os.getenv("MODEL_ID", "nvidia/nemotron-3-super-120b-a12b:free")
    
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

#######------------- Prompt Template and Chain-------------#######

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

#######------------- Speech2text and Pipeline-------------#######

# Speech-to-text pipeline
def transcript_audio(audio_file):
    pipe = pipeline(
        "automatic-speech-recognition",
        model="./whisper-tiny.en",  # chemin local
        chunk_length_s=30,
    )
    raw_transcript = pipe(audio_file, batch_size=8)["text"]
    ascii_transcript = remove_non_ascii(raw_transcript)

    adjusted_transcript = product_assistant(ascii_transcript)
    result = chain.invoke({"context": adjusted_transcript})

    # Write the result to a file for downloading
    output_file = "meeting_minutes_and_tasks.txt"
    with open(output_file, "w") as file:
        file.write(result)

    # Return the textual result and the file for download
    return result, output_file

#######------------- Gradio Interface-------------#######

audio_input = gr.Audio(sources="upload", type="filepath", label="Upload your audio file")
output_text = gr.Textbox(label="Meeting Minutes and Tasks")
download_file = gr.File(label="Download the Generated Meeting Minutes and Tasks")

iface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=[output_text, download_file],
    title="AI Meeting Assistant",
    description="Upload an audio file of a meeting. This tool will transcribe the audio, fix product-related terminology, and generate meeting minutes along with a list of tasks."
)

iface.launch(server_name="0.0.0.0", server_port=5000)
