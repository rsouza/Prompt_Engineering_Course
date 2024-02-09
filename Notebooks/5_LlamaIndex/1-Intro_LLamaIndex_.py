# Databricks notebook source
# MAGIC %md
# MAGIC ### Quick intro to LlamaIndex  
# MAGIC Sources: [1](https://lmy.medium.com/comparing-langchain-and-llamaindex-with-4-tasks-2970140edf33), [2](https://docs.llamaindex.ai/en/stable/), [3](https://github.com/run-llama/llama_index), [4](https://nanonets.com/blog/llamaindex/)  
# MAGIC
# MAGIC LlamaIndex is a "data framework" to help you build LLM apps. It provides the following tools:
# MAGIC
# MAGIC + Offers data connectors to ingest your existing data sources and data formats (APIs, PDFs, docs, SQL, etc.).
# MAGIC + Provides ways to structure your data (indices, graphs) so that this data can be easily used with LLMs.
# MAGIC + Provides an advanced retrieval/query interface over your data: Feed in any LLM input prompt, get back retrieved context and knowledge-augmented output.
# MAGIC + Allows easy integrations with your outer application framework (e.g. with LangChain, Flask, Docker, ChatGPT, anything else).
# MAGIC + LlamaIndex provides tools for both beginner users and advanced users.  
# MAGIC
# MAGIC The high-level API allows beginner users to use LlamaIndex to ingest and query their data in 5 lines of code.  
# MAGIC The lower-level APIs allow advanced users to customize and extend any module (data connectors, indices, retrievers, query engines, reranking modules), to fit their needs.  
# MAGIC
# MAGIC LlamaIndex provides the following tools:
# MAGIC + Data connectors ingest your existing data from their native source and format. These could be APIs, PDFs, SQL, and (much) more.
# MAGIC + Data indexes structure your data in intermediate representations that are easy and performant for LLMs to consume.
# MAGIC + Engines provide natural language access to your data. For example:
# MAGIC + Query engines are powerful retrieval interfaces for knowledge-augmented output.
# MAGIC + Chat engines are conversational interfaces for multi-message, “back and forth” interactions with your data.
# MAGIC + Data agents are LLM-powered knowledge workers augmented by tools, from simple helper functions to API integrations and more.
# MAGIC + Application integrations tie LlamaIndex back into the rest of your ecosystem. This could be LangChain, Flask, Docker, ChatGPT, or… anything else!  

# COMMAND ----------

# DBTITLE 0,ro
!pip install -q openai==0.27.0
#!pip install -qU llama-index            # Just the core components
!pip install llama-index[local_models] # Installs tools useful for private LLMs, local inference, and HuggingFace models
#!pip install llama-index[postgres]     # Is useful if you are working with Postgres, PGVector or Supabase
#!pip install llama-index[query_tools]  # Gives you tools for hybrid search, structured outputs, and node post-processing
# !pip install google-generativeai  #PALM
!pip install -qU pypdf
!pip install -qU docx2txt
!pip install -qU sentence-transformers
dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
import shutil
import glob
import logging
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
#import tiktoken
#from funcy import lcat, lmap, linvoke
#from IPython.display import display, Markdown
import openai

#OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]  #It has to be defined before importing LlamaIndex modules

## LlamaIndex LLMs
#from openai import OpenAI
#from openai import AzureOpenAI
from llama_index.llms import AzureOpenAI
from llama_index.llms import ChatMessage
#from llama_index.llms import Ollama
#from llama_index.llms import PaLM

## LlamaIndex Embeddings
from llama_index.embeddings import OpenAIEmbedding
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index.embeddings import resolve_embed_model

## Llamaindex readers 
from llama_index import SimpleDirectoryReader
#from llama_index import ResponseSynthesizer

## LlamaIndex Index Types
#from llama_index import GPTListIndex             
from llama_index import VectorStoreIndex
#from llama_index import GPTVectorStoreIndex  
#from llama_index import GPTTreeIndex
#from llama_index import GPTKeywordTableIndex
#from llama_index import GPTSimpleKeywordTableIndex
#from llama_index import GPTDocumentSummaryIndex
#from llama_index import GPTKnowledgeGraphIndex
#from llama_index.indices.struct_store import GPTPandasIndex
#from llama_index.vector_stores import ChromaVectorStore

## LlamaIndex Context Managers
from llama_index import ServiceContext
from llama_index import StorageContext
from llama_index import load_index_from_storage
from llama_index import set_global_service_context
from llama_index.response_synthesizers import get_response_synthesizer
#from llama_index import LLMPredictor

## LlamaIndex Callbacks
#from llama_index.callbacks import CallbackManager
#from llama_index.callbacks import LlamaDebugHandler


## Defining LLM Model
llm_option = "OpenAI"
if llm_option == "OpenAI":
    openai.api_type = "azure"
    azure_endpoint = "https://rg-rbi-aa-aitest-dsacademy.openai.azure.com/"
    #azure_endpoint = "https://chatgpt-summarization.openai.azure.com/"
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
elif llm_option == "Local":  #https://docs.llamaindex.ai/en/stable/module_guides/models/llms.html and https://docs.llamaindex.ai/en/stable/module_guides/models/llms/local.html
    print("Make sure you have installed Local Models - !pip install llama-index[local_models]")
    llm = Ollama(model="mistral", request_timeout=30.0)
else:
    raise ValueError("Invalid LLM Model")

## Defining Embedding Model
emb_option = "Local"
if emb_option == "OpenAI":
    embed_model_name = "text-embedding-ada-002"
    embed_model_deployment_name = "model-text-embedding-ada-002"
    embed_model = AzureOpenAIEmbedding(model=embed_model_name,
                                       deployment_name=embed_model_deployment_name,
                                       api_key=openai.api_key,
                                       azure_endpoint=azure_endpoint)
elif emb_option == "Local":
    embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")   ## bge-m3 embedding model
else:
    raise ValueError("Invalid Embedding Model")

## Logging Optionals
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Text Completion Example

# COMMAND ----------

resp = llm.complete("Paul Graham is ")
print(resp)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Chat Example Example

# COMMAND ----------

messages = [ChatMessage(role="system", content="You are a pirate with a colorful personality"),
            ChatMessage(role="user", content="What is your name"),
            ]
resp = llm.chat(messages)
print(resp)

# COMMAND ----------

# MAGIC %md ### Quickstart: Implementing a RAG Pipeline:
# MAGIC
# MAGIC ![](https://docs.llamaindex.ai/en/stable/_images/basic_rag.png)

# COMMAND ----------

# MAGIC %md ### Examining Documents Folder

# COMMAND ----------

DOCS_DIR = "../../Data/txt/"
docs = os.listdir(DOCS_DIR)
docs = [d for d in docs] # if d.endswith(".txt")]
docs.sort()
for doc in docs:
    print(doc)

# COMMAND ----------

# MAGIC %md #### Setting the Service Context

# COMMAND ----------

service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)
set_global_service_context(service_context)

# COMMAND ----------

# MAGIC %md ### Creating the Vector Store

# COMMAND ----------

PERSIST_DIR = "/Workspace/ds-academy-research/LLamaIndex_quick/"

# COMMAND ----------

if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)
print(f"Creating Directory {PERSIST_DIR}")
os.mkdir(PERSIST_DIR)

if os.listdir(PERSIST_DIR) == []:
    print("Loading Documents...")
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    print("Creating Vector Store...")
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    print("Persisting Vector Store...")
    index.storage_context.persist(persist_dir=PERSIST_DIR)

# COMMAND ----------

# MAGIC %md ### Reading from existing Vector Store

# COMMAND ----------

print("Reading from Vector Store...")
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)

# COMMAND ----------

index.ref_doc_info

# COMMAND ----------

# MAGIC %md ### Querying your data  

# COMMAND ----------

query_engine = index.as_query_engine(retriever_mode="embedding", response_mode="accumulate", verbose=True)

# COMMAND ----------

response = query_engine.query("Who was Romeo?")
print(response)

# COMMAND ----------

response = query_engine.query("Who did the proofreading of Pride and Prejudice?")
print(response)

# COMMAND ----------

response = query_engine.query("Who is the publisher of Pride and Prejudice?")
print(response)

# COMMAND ----------

# MAGIC %md ### Chat with your Data  
# MAGIC
# MAGIC [Available Chat Modes](https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/usage_pattern.html)  
# MAGIC + best - Turn the query engine into a tool, for use with a ReAct data agent or an OpenAI data agent, depending on what your LLM supports. OpenAI data agents require gpt-3.5-turbo or gpt-4 as they use the function calling API from OpenAI.
# MAGIC + condense_question - Look at the chat history and re-write the user message to be a query for the index. Return the response after reading the response from the query engine.
# MAGIC + context - Retrieve nodes from the index using every user message. The retrieved text is inserted into the system prompt, so that the chat engine can either respond naturally or use the context from the query engine.
# MAGIC + condense_plus_context - A combination of condense_question and context. Look at the chat history and re-write the user message to be a retrieval query for the index. The retrieved text is inserted into the system prompt, so that the chat engine can either respond naturally or use the context from the query engine.
# MAGIC + simple - A simple chat with the LLM directly, no query engine involved.
# MAGIC + react - Same as best, but forces a ReAct data agent.
# MAGIC + openai - Same as best, but forces an OpenAI data agent.

# COMMAND ----------

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
chat_engine.reset()

# COMMAND ----------

response = chat_engine.chat("What did happen to Romeo?")
print(response)
response = chat_engine.chat("When was that?")
print(response)
response = chat_engine.chat("What kind of poison?")
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using  REPL interface  

# COMMAND ----------

chat_engine.chat_repl()
