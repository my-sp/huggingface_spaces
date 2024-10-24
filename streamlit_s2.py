import streamlit as st
import transformers
import torch

HF_TOKEN=st.secrets["HF_Token"]
# Load the model and pipeline
model_id = "meta-llama/Llama-3.1-8B-Instruct"

# Set up the pipeline with the Hugging Face token
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16, "use_auth_token": HF_TOKEN}
)

# Streamlit user interface
st.title("LLM Model Inference")
input_text = st.text_input("Enter your prompt:")

if st.button("Generate"):
    if input_text:  # Check if the input is not empty
        # Generate text using the pipeline
        messages = [
        {"role": "system", "content": "You are a question answering assistant."},
        {"role": "user", "content": input_text}
        ]
        response = pipeline(messages, max_new_tokens=30)  
        st.write("Generated Response:")
        st.write(response[0]['generated_text'][-1]['content'])  
    else:
        st.error("Please enter a prompt to generate text.")
