import streamlit as st
import requests
from huggingface_hub import InferenceClient

st.title("Streamlit with Hugging Face API")

def get_models():
     return ["meta-llama/Meta-Llama-3-8B-Instruct","meta-llama/Llama-3.1-8B-Instruct"]

models = get_models()
model_names = get_models()
placeholder_option = "Select a model"
model_names_with_placeholder = [placeholder_option] + model_names
selected_model = st.selectbox("Select the model:", model_names_with_placeholder)
prompt1 = st.chat_input("Message")

if selected_model != placeholder_option:
    st.write(f"You selected: {selected_model}")
    api_token = "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"  
    client = InferenceClient(model=selected_model, token=api_token)

    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt1:  
        with st.chat_message("user"):
            st.markdown(prompt1)
        st.session_state.messages.append({"role": "user", "content": prompt1})
        try:
            response = client.text_generation(prompt1)
        except Exception as e:
            response = f"An error occurred: {e}"
        with st.chat_message("assistant"):
            st.markdown(response)
    # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.write("Please select a model.")
