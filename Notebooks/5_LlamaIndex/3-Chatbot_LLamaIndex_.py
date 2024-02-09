# Databricks notebook source
# MAGIC %md
# MAGIC ### Intro to LlamaIndex  
# MAGIC Sources: [1](https://lmy.medium.com/comparing-langchain-and-llamaindex-with-4-tasks-2970140edf33), [2](https://docs.llamaindex.ai/en/stable/), [3](https://github.com/run-llama/llama_index), [4](https://nanonets.com/blog/llamaindex/)  
# MAGIC
# MAGIC #### Retrieval Augmented Generation (RAG)
# MAGIC LLMs are trained on enormous bodies of data but they aren’t trained on your data. Retrieval-Augmented Generation (RAG) solves this problem by adding your data to the data LLMs already have access to. You will see references to RAG frequently in this documentation.  
# MAGIC In RAG, your data is loaded and prepared for queries or “indexed”. User queries act on the index, which filters your data down to the most relevant context. This context and your query then go to the LLM along with a prompt, and the LLM provides a response.  
# MAGIC Even if what you’re building is a chatbot or an agent, you’ll want to know RAG techniques for getting data into your application.  
# MAGIC
# MAGIC #### Stages within RAG
# MAGIC There are five key stages within RAG, which in turn will be a part of any larger application you build. These are:
# MAGIC + Loading: this refers to getting your data from where it lives – whether it’s text files, PDFs, another website, a database, or an API – into your pipeline. LlamaHub provides hundreds of connectors to choose from.
# MAGIC + Indexing: this means creating a data structure that allows for querying the data. For LLMs this nearly always means creating vector embeddings, numerical representations of the meaning of your data, as well as numerous other metadata strategies to make it easy to accurately find contextually relevant data.
# MAGIC + Storing: once your data is indexed you will almost always want to store your index, as well as other metadata, to avoid having to re-index it.
# MAGIC + Querying: for any given indexing strategy there are many ways you can utilize LLMs and LlamaIndex data structures to query, including sub-queries, multi-step queries and hybrid strategies.
# MAGIC + Evaluation: a critical step in any pipeline is checking how effective it is relative to other strategies, or when you make changes. Evaluation provides objective measures of how accurate, faithful and fast your responses to queries are.
# MAGIC
# MAGIC #### Important concepts within each step
# MAGIC There are also some terms you’ll encounter that refer to steps within each of these stages.  
# MAGIC + Loading stage
# MAGIC **Nodes** and **Documents**: A Document is a container around any data source - for instance, a PDF, an API output, or retrieve data from a database.  
# MAGIC A Node is the atomic unit of data in LlamaIndex and represents a “chunk” of a source Document. Nodes have metadata that relate them to the document they are in and to other nodes.  
# MAGIC **Connectors**: A data connector (often called a Reader) ingests data from different data sources and data formats into Documents and Nodes.  
# MAGIC
# MAGIC + Indexing Stage  
# MAGIC **Indexes**: Once you’ve ingested your data, LlamaIndex will help you index the data into a structure that’s easy to retrieve. This usually involves generating vector embeddings which are stored in a specialized database called a vector store. Indexes can also store a variety of metadata about your data.  
# MAGIC **Embeddings** LLMs generate numerical representations of data called embeddings. When filtering your data for relevance, LlamaIndex will convert queries into embeddings, and your vector store will find data that is numerically similar to the embedding of your query.  
# MAGIC
# MAGIC + Querying Stage
# MAGIC **Retrievers**: A retriever defines how to efficiently retrieve relevant context from an index when given a query. Your retrieval strategy is key to the relevancy of the data retrieved and the efficiency with which it’s done.  
# MAGIC **Routers**: A router determines which retriever will be used to retrieve relevant context from the knowledge base. More specifically, the RouterRetriever class, is responsible for selecting one or multiple candidate retrievers to execute a query. They use a selector to choose the best option based on each candidate’s metadata and the query.  
# MAGIC Node Postprocessors: A node postprocessor takes in a set of retrieved nodes and applies transformations, filtering, or re-ranking logic to them.  
# MAGIC Response Synthesizers: A response synthesizer generates a response from an LLM, using a user query and a given set of retrieved text chunks.  
# MAGIC
# MAGIC #### Putting it all together
# MAGIC There are endless use cases for data-backed LLM applications but they can be roughly grouped into three categories:
# MAGIC
# MAGIC + Query Engines: A query engine is an end-to-end pipeline that allows you to ask questions over your data. It takes in a natural language query, and returns a response, along with reference context retrieved and passed to the LLM.
# MAGIC + Chat Engines: A chat engine is an end-to-end pipeline for having a conversation with your data (multiple back-and-forth instead of a single question-and-answer).
# MAGIC + Agents: An agent is an automated decision-maker powered by an LLM that interacts with the world via a set of tools. Agents can take an arbitrary number of steps to complete a given task, dynamically deciding on the best course of action rather than following pre-determined steps. This gives it additional flexibility to tackle more complex tasks.  

# COMMAND ----------

# MAGIC %md #### Installing Packages

# COMMAND ----------

# DBTITLE 0,ro
#!pip install -q openai==0.27.0
!pip install -qU llama-index            # Just the core components
#!pip install -qU llama-index[local_models] # Installs tools useful for private LLMs, local inference, and HuggingFace models
#!pip install -q llama-index[postgres]     # Is useful if you are working with Postgres, PGVector or Supabase
#!pip install -q llama-index[query_tools]  # Gives you tools for hybrid search, structured outputs, and node post-processing
!pip install -q llama-hub 
#!pip install -qU chromadb
!pip install -qU pypdf
!pip install -qU docx2txt
!pip install -qU sentence-transformers
!pip install -q unstructured
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md #### Importing Packages

# COMMAND ----------

import os
import sys
import shutil
import glob
import logging
from pathlib import Path
import nest_asyncio
nest_asyncio.apply()

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
#import tiktoken
#from funcy import lcat, lmap, linvoke
#from IPython.display import Markdown, display
import openai
#import chromadb

#OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]  #It has to be defined before importing LlamaIndex modules

## LlamaIndex LLMs
#from openai import OpenAI
#from openai import AzureOpenAI
from llama_index.llms import AzureOpenAI
#from llama_index.llms import Ollama
#from llama_index.llms import PaLM

## LlamaIndex Embeddings
from llama_index.embeddings import OpenAIEmbedding
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index.embeddings import resolve_embed_model

## Llamaindex readers 
#from llama_index import SimpleDirectoryReader
from llama_hub.file.unstructured.base import UnstructuredReader

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
from llama_index.response_synthesizers import ResponseMode
from llama_index.schema import Node
#from llama_index import LLMPredictor

## LlamaIndex Tools
from llama_index.tools import QueryEngineTool
from llama_index.tools import ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.chat_engine import SimpleChatEngine

## LlamaIndex Agents
from llama_index.agent import OpenAIAgent

## LlamaIndex Callbacks
from llama_index.callbacks import CallbackManager
from llama_index.callbacks import LlamaDebugHandler

# COMMAND ----------

# MAGIC %md #### Defining Model and Endpoints

# COMMAND ----------

## Defining LLM Model
## A full guide to using and configuring LLMs available here: https://docs.llamaindex.ai/en/stable/module_guides/models/llms.html
## Check also: https://docs.llamaindex.ai/en/stable/module_guides/models/llms/local.html
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
elif llm_option == "Local":  
    print("Make sure you have installed Local Models - !pip install llama-index[local_models]")
    llm = Ollama(model="mistral", request_timeout=30.0)
else:
    raise ValueError("Invalid LLM Model")

## Defining Embedding Model
## A full guide to using and configuring embedding models is available here. https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html
emb_option = "OpenAI"
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

PERSIST_DIR = "/Workspace/ds-academy-research/LLamaIndex/VectorStoreIndex/"

# COMMAND ----------

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

service_context = ServiceContext.from_defaults(llm=llm,
                                               #prompt_helper= ,
                                               embed_model=embed_model,
                                               #node_parser= ,
                                               #chunk_size=1000,                                        #Parse Documents into smaller chunks
                                               callback_manager=callback_manager,                       #Visualize execution
                                               #system_prompt=(Optional[str]),                          #System-wide prompt to be prepended to all input prompts, used to guide system “decision making”
                                               #query_wrapper_prompt=(Optional[BasePromptTemplate]),    #A format to wrap passed-in input queries.
                                               )

set_global_service_context(service_context)

# COMMAND ----------

# MAGIC %md #### [Storage Context](https://docs.llamaindex.ai/en/stable/api_reference/storage.html)  
# MAGIC LlamaIndex offers core abstractions around storage of Nodes, indices, and vectors. A key abstraction is the StorageContext - this contains the underlying BaseDocumentStore (for nodes), BaseIndexStore (for indices), and VectorStore (for vectors).
# MAGIC StorageContext defines the storage backend for where the documents, embeddings, and indexes are stored.   
# MAGIC ```
# MAGIC storage_context = StorageContext.from_defaults(persist_dir="<path/to/index>")
# MAGIC ```
# MAGIC You can learn more about [storage](https://docs.llamaindex.ai/en/stable/module_guides/storing/storing.html) and how to [customize](https://docs.llamaindex.ai/en/stable/module_guides/storing/customization.html) it.  

# COMMAND ----------

# MAGIC %md ### Reading [Vector Store Index](https://docs.llamaindex.ai/en/stable/api_reference/query/retrievers/vector_store.html)  

# COMMAND ----------

vectorstoreindex = load_index_from_storage(storage_context=StorageContext.from_defaults(persist_dir=PERSIST_DIR))

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Querying Index

# COMMAND ----------

query_engine = vectorstoreindex.as_query_engine(retriever_mode="embedding",
                                                response_mode="compact",
                                                verbose=True)
response = query_engine.query("Will GenAI create new jobs?")
print(response)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Creating an Simple Interactive Chatbot for our Index

# COMMAND ----------

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
chat_engine.reset()
chat_engine.chat_repl()

# COMMAND ----------

# MAGIC %md
# MAGIC https://sharmadave.medium.com/llama-index-unleashes-the-power-of-chatgpt-over-your-own-data-b67cc2e4e277  
# MAGIC https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/  
# MAGIC https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/chatbots/building_a_chatbot.html

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Creating an Customized Chatbot for our Index

# COMMAND ----------

def create_custom_chatEngine(index):
   
    template = (
    "Following Informations : \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Please answer the question always from the first person perspective and always start your answer with Renato: {query_str}\n"
)
    qa_template = Prompt(template)
    query_engine = index.as_query_engine(text_qa_template=qa_template)
    chat_engine = CondenseQuestionChatEngine.from_defaults(query_engine=query_engine, verbose=False)
    return chat_engine

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Creating an [interactive Chatbot](https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/chatbots/building_a_chatbot.html) with Agents

# COMMAND ----------

years = [2022, 2021, 2020, 2019]
loader = UnstructuredReader()
doc_set = {}
all_docs = []
for year in years:
    year_docs = loader.load_data(file=Path(f"../../Data/html/UBER_{year}.html"), split_documents=False)
    for d in year_docs:
        d.metadata = {"year": year}
    doc_set[year] = year_docs
    all_docs.extend(year_docs)

index_set = {}
service_context = ServiceContext.from_defaults(chunk_size=512)
for year in years:
    storage_context = StorageContext.from_defaults()
    cur_index = VectorStoreIndex.from_documents(doc_set[year],
                                                service_context=service_context,
                                                storage_context=storage_context,
                                                )
    index_set[year] = cur_index
    storage_context.persist(persist_dir=f"./storage/{year}")

index_set = {}
for year in years:
    storage_context = StorageContext.from_defaults(persist_dir=f"./storage/{year}")
    cur_index = load_index_from_storage(storage_context, 
                                        service_context=service_context,
                                        )
    index_set[year] = cur_index


individual_query_engine_tools = [QueryEngineTool(query_engine=index_set[year].as_query_engine(),
                                                 metadata=ToolMetadata(name=f"vector_index_{year}", description=f"useful for when you want to answer queries about the {year} SEC 10-K for Uber",),
                                                 ) for year in years]    

query_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=individual_query_engine_tools,
                                                    service_context=service_context,)

query_engine_tool = QueryEngineTool(query_engine=query_engine,
                                    metadata=ToolMetadata(name="sub_question_query_engine",
                                                          description="useful for when you want to answer queries that require analyzing multiple SEC 10-K documents for Uber",
                                                          ),
                                    )

tools = individual_query_engine_tools + [query_engine_tool]
agent = OpenAIAgent.from_tools(tools, verbose=True)

# COMMAND ----------

response = agent.chat("hi, i am bob")
print(str(response))

# COMMAND ----------

response = agent.chat("What were some of the biggest risk factors in 2020 for Uber?")
print(str(response))

# COMMAND ----------

cross_query_str = "Compare/contrast the risk factors described in the Uber 10-K across years. Give answer in bullet points."
response = agent.chat(cross_query_str)
print(str(response))

# COMMAND ----------

agent = OpenAIAgent.from_tools(tools)  # verbose=False by default

while True:
    text_input = input("User: ")
    if text_input == "exit":
        break
    response = agent.chat(text_input)
    print(f"Agent: {response}")
