import subprocess
import sys
import os
import sys
import shutil
import glob
import re
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import openai
import streamlit as st
from llama_index.llms import AzureOpenAI
from llama_index.llms import ChatMessage
from llama_index.llms import MessageRole
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index import ServiceContext
from llama_index import StorageContext
from llama_index import load_index_from_storage
from llama_index import set_global_service_context
#from aa_llm_utils.utils import ensure_certificates
#ensure_certificates()

openai.api_type = "azure"
azure_endpoint = "https://rg-rbi-aa-aitest-dsacademy.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.environ["OPENAI_API_KEY"]
deployment_name = "model-gpt-35-turbo"
openai_model_name = "gpt-35-turbo"
llm = AzureOpenAI(api_key=openai.api_key,
                  azure_endpoint=azure_endpoint,
                  model=openai_model_name,
                  engine=deployment_name,
                  api_version=openai.api_version,
                  )

embed_model_name = "text-embedding-ada-002"
embed_model_deployment_name = "model-text-embedding-ada-002"
embed_model = AzureOpenAIEmbedding(model=embed_model_name,
                                   deployment_name=embed_model_deployment_name,
                                   api_key=openai.api_key,
                                   azure_endpoint=azure_endpoint)

PERSIST_DIR = "/Workspace/ds-academy-research/LLamaIndex/VectorStoreIndex/"
service_context = ServiceContext.from_defaults(llm=llm,
                                               embed_model=embed_model,
                                               )
set_global_service_context(service_context)

vectorstoreindex = load_index_from_storage(storage_context=StorageContext.from_defaults(persist_dir=PERSIST_DIR))
chat_engine = vectorstoreindex.as_chat_engine(chat_mode="condense_question", verbose=True)

st.header("Chat with Documents from your Vector Index")
# Initialize the chat message history
if "messages" not in st.session_state.keys(): 
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Large Language Models!"}
    ]
# Prompt for user input and save to chat history    
if prompt := st.chat_input("Your question"): 
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display the prior chat messages
for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            # Add response to message history
            st.session_state.messages.append(message) 