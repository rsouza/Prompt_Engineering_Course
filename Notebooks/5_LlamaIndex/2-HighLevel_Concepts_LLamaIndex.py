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
#!pip install -qU chromadb
!pip install -qU pypdf
!pip install -qU docx2txt
!pip install -qU sentence-transformers
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
from llama_index import SimpleDirectoryReader

## LlamaIndex Index Types
from llama_index import GPTListIndex             
from llama_index import VectorStoreIndex
from llama_index import GPTVectorStoreIndex  
from llama_index import GPTTreeIndex
from llama_index import GPTKeywordTableIndex
from llama_index import GPTSimpleKeywordTableIndex
from llama_index import GPTDocumentSummaryIndex
from llama_index import GPTKnowledgeGraphIndex
from llama_index.indices.struct_store import GPTPandasIndex
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

DOCS_DIR = "/Workspace/ds-academy-research/Docs/LLM_sample/"
PERSIST_DIR = "/Workspace/ds-academy-research/LLamaIndex/"

# COMMAND ----------

# MAGIC %md #### [ServiceContext](https://docs.llamaindex.ai/en/stable/api_reference/service_context.html) defines a handful of services and configurations used across a LlamaIndex pipeline.  
# MAGIC + Embeddings  
# MAGIC + OpenAIEmbedding  
# MAGIC + HuggingFaceEmbedding
# MAGIC + OptimumEmbedding
# MAGIC + InstructorEmbedding
# MAGIC + LangchainEmbedding
# MAGIC + GoogleUnivSentEncoderEmbedding
# MAGIC + [Node Parsers](https://docs.llamaindex.ai/en/stable/api_reference/service_context/node_parser.html)
# MAGIC + PromptHelper
# MAGIC + LLMs

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

# MAGIC %md ### Customizing a RAG Pipeline:
# MAGIC
# MAGIC ![](https://docs.llamaindex.ai/en/stable/_images/basic_rag.png)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Creating the Index  
# MAGIC
# MAGIC LlamaIndex is known for offering different types of indexes; each one is more suited to a different purpose.
# MAGIC
# MAGIC + #### A) List Index  
# MAGIC The list index is a simple data structure where nodes are stored in a sequence.  
# MAGIC The document texts are chunked up, converted to nodes, and stored in a list during index construction.
# MAGIC
# MAGIC ![](https://miro.medium.com/v2/resize:fit:720/format:webp/0*rBBHy019pbV9kyxh.png)
# MAGIC ![](https://miro.medium.com/v2/resize:fit:720/format:webp/0*8ANcn6OBBVzIHAd0.png)
# MAGIC ![](https://miro.medium.com/v2/resize:fit:720/format:webp/0*NQAUXYHPq0wh8zhw.png)
# MAGIC ***
# MAGIC + #### B) Vector Store Index  
# MAGIC It is most common and simple to use, allows answering a query over a large corpus of data  
# MAGIC By default, GPTVectorStoreIndex uses an in-memory SimpleVectorStore that’s initialized as part of the default storage context.  
# MAGIC Unlike list index, vector-store based indices generate embeddings during index construction
# MAGIC
# MAGIC ![](https://miro.medium.com/v2/resize:fit:720/format:webp/0*IbHJovGnj38dDHsB.png)
# MAGIC ![](https://miro.medium.com/v2/resize:fit:720/format:webp/0*-9QtrMEBYrAFWDMH.png)
# MAGIC ***
# MAGIC + #### C) Tree Index
# MAGIC It is useful for summarizing a collection of documents  
# MAGIC The tree index is a tree-structured index, where each node is a summary of the children's nodes.  
# MAGIC During index construction, the tree is constructed in a bottoms-up fashion until we end up with a set of root nodes.  
# MAGIC The tree index builds a hierarchical tree from a set of Nodes (which become leaf nodes in this tree).  
# MAGIC Unlike vector index, LlamaIndex won’t call LLM to generate embedding but will generate it during query time.   
# MAGIC Embeddings are lazily generated and then cached (if retriever_mode="embedding" is specified during query(...)), and not during index construction.  
# MAGIC
# MAGIC ![](https://miro.medium.com/v2/resize:fit:720/format:webp/0*906uyjc0HBDfiyzw.png)
# MAGIC ![](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*CpUvD5VejES-JdRq.png)
# MAGIC ***
# MAGIC + #### D) Keyword Table Index
# MAGIC It is useful for routing queries to the disparate data source  
# MAGIC The keyword table index extracts keywords from each Node and builds a mapping from each keyword to the corresponding Nodes of that keyword.  
# MAGIC During query time, we extract relevant keywords from the query and match those with pre-extracted Node keywords to fetch the corresponding Nodes. The extracted Nodes are passed to the Response Synthesis module.  
# MAGIC GPTKeywordTableIndex use LLM to extract keywords from each document, meaning it do require LLM calls during build time. However, if you use GPTSimpleKeywordTableIndex which uses a regex keyword extractor to extract keywords from each document, it won’t call LLM during build time
# MAGIC
# MAGIC ![](https://miro.medium.com/v2/resize:fit:720/format:webp/0*DUR4yHaMam-vln3t.png)
# MAGIC ![](https://miro.medium.com/v2/resize:fit:720/format:webp/0*ERSNFpKoKfbIICkz.png)
# MAGIC ***
# MAGIC + #### E) Document Summary Index
# MAGIC This index can extract and index an unstructured text summary for each document, which enhances retrieval performance beyond existing approaches. It contains more information than a single text chunk and carries more semantic meaning than keyword tags. It also allows for flexible retrieval, including both LLM and embedding-based approaches. During build-time, this index ingests document and use LLM to extract a summary from each document. During query time, it retrieves relevant documents to query based on summaries using the following approaches:
# MAGIC + LLM-based Retrieval: get collections of document summaries and request LLM to identify the relevant documents + relevance score
# MAGIC + Embedding-based Retrieval: utilize summary embedding similarity to retrieve relevant documents, and impose a top-k limit to the number of retrieved results.
# MAGIC
# MAGIC
# MAGIC ![](https://miro.medium.com/v2/resize:fit:720/format:webp/0*Sr1_53f_HAXwbsQ5.png)
# MAGIC ***
# MAGIC + #### F) Graph Index
# MAGIC It is useful for building a knowledge graph  
# MAGIC It builds the index by extracting knowledge triples in the form (subject, predicate, object) over a set of docs. During the query time, it can either query using just the knowledge graph as context or leverage the underlying text from each entity as context. By leveraging the underlying text, we can ask more complicated queries with respect to the contents of the document.  
# MAGIC With LlamaIndex, you have the ability to create composite indices by building indices on top of existing ones. This feature empowers you to efficiently index your complete document hierarchy and provide tailored knowledge to GPT.
# MAGIC By leveraging composability, you can define indices at multiple levels, such as lower-level indices for individual documents and higher-level indices for groups of documents.  
# MAGIC
# MAGIC ![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*rEg1wqA7V7HXUWy4LP6zXQ.png)
# MAGIC ***
# MAGIC + #### Pandas and SQL Indexes
# MAGIC Useful for structured data.
# MAGIC ***
# MAGIC ##### Summarizing
# MAGIC There are important aspects to regard, namely: indexation cost, and indexation time (speed).
# MAGIC + Indexing Cost: The expense of indexing is a crucial factor to consider. This is particularly significant when dealing with massive datasets.  
# MAGIC + Indexing Speed: The second important issue is the time of document indexing, i.e. preparing the entire solution for operation. Indexation time varies but it is a one-off and also depends on the OpenAI server.  
# MAGIC Usually, the pdf with 40 pages will take approximately 5 seconds. Imagine a huge dataset with more than 100k pages, it could take to several days. We can leverage the async method to reduce the indexing time.  
# MAGIC
# MAGIC ![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*cyRHH_0z39JmFGeLYBWFEA.png)
# MAGIC
# MAGIC Sources: [1](https://betterprogramming.pub/llamaindex-how-to-use-index-correctly-6f928b8944c6), [2](https://docs.llamaindex.ai/en/stable/module_guides/indexing/indexing.html), [3](https://mikulskibartosz.name/llama-index-which-index-should-you-use)

# COMMAND ----------

# MAGIC %md #### [Storage Context](https://docs.llamaindex.ai/en/stable/api_reference/storage.html)  
# MAGIC LlamaIndex offers core abstractions around storage of Nodes, indices, and vectors. A key abstraction is the StorageContext - this contains the underlying BaseDocumentStore (for nodes), BaseIndexStore (for indices), and VectorStore (for vectors).
# MAGIC StorageContext defines the storage backend for where the documents, embeddings, and indexes are stored.   
# MAGIC You can learn more about [storage](https://docs.llamaindex.ai/en/stable/module_guides/storing/storing.html) and how to [customize](https://docs.llamaindex.ai/en/stable/module_guides/storing/customization.html) it.  

# COMMAND ----------

storage_context = StorageContext.from_defaults()

# COMMAND ----------

# MAGIC %md #### Deleting existing Indexes  
# MAGIC
# MAGIC (Only if you want to recreate all indexes)

# COMMAND ----------

#if not os.path.exists(PERSIST_DIR):
#    print(f"Creating Directory {PERSIST_DIR}")
#    os.mkdir(PERSIST_DIR)
#else:
#    print(f"Re-Creating Directory {PERSIST_DIR}")
#    shutil.rmtree(PERSIST_DIR)
#    os.mkdir(PERSIST_DIR)

# COMMAND ----------

# MAGIC %md  
# MAGIC #### Some remarks:
# MAGIC + We will load documens from a directory, but you can check all integrations (readers) [here](https://llamahub.ai/?tab=loaders)  
# MAGIC + We could also transform documents in nodes and create the index directly from [nodes](https://docs.llamaindex.ai/en/stable/api_reference/service_context/node_parser.html)  
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### A) Creating (or loading) List Index  
# MAGIC
# MAGIC The GPTListIndex index is perfect when you don’t have many documents. Instead of trying to find the relevant data, the index concatenates all chunks and sends them all to the LLM. If the resulting text is too long, the index splits the text and asks LLM to refine the answer.  
# MAGIC GPTListIndex may be a good choice when we have a few questions to answer using a handful of documents. It may give us the best answer because AI will get all the available data, but it is also quite expensive. We pay per token, so sending all the documents to the LLM may not be the best idea.  
# MAGIC

# COMMAND ----------

LISTINDEXDIR = PERSIST_DIR + 'ListIndex' 

if not os.path.exists(LISTINDEXDIR) or os.listdir(LISTINDEXDIR) == []:
    os.mkdir(LISTINDEXDIR)
    print("Loading Documents...")
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    print("Creating Index...")
    listindex = GPTListIndex.from_documents(documents, service_context=service_context)               ## List Index
    print("Persisting Index...")
    listindex.storage_context.persist(persist_dir=LISTINDEXDIR)
else:
    print("Reading from Index...")
    listindex = load_index_from_storage(storage_context)

# COMMAND ----------

# MAGIC %md
# MAGIC #### B) Creating (or loading) Vector Store Index  
# MAGIC
# MAGIC By default, LlamaIndex uses a simple in-memory vector store, but you can use [another solution](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores.html)   
# MAGIC VectorStoreIndex creates numerical vectors from the text using word embeddings and retrieves relevant documents based on the similarity of the vectors.  
# MAGIC When we index the documents, the library chunks them into a number of nodes and calls the embeddings endpoint of OpenAI API by default.  
# MAGIC We can also reuse the index to answer multiple questions without sending the documents to LLM many times. The disadvantage is that the quality of the answers depends on the quality of the embeddings. If the embeddings are not good enough, the LLM will not be able to generate a good response. (e.g. skip the table of contents when you index the documents)  

# COMMAND ----------

VECTORINDEXDIR = PERSIST_DIR + 'VectorStoreIndex' 

if not os.path.exists(VECTORINDEXDIR) or os.listdir(VECTORINDEXDIR) == []:
    os.mkdir(VECTORINDEXDIR)
    print("Loading Documents...")
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    print("Creating Index...")
    vectorstoreindex = VectorStoreIndex.from_documents(documents, service_context=service_context)           ## Vector Store Index
    print("Persisting Index...")
    vectorstoreindex.storage_context.persist(persist_dir=VECTORINDEXDIR)
else:
    print("Reading from Index...")
    vectorstoreindex = load_index_from_storage(storage_context)

# COMMAND ----------

# MAGIC %md
# MAGIC #### C) Creating (or loading) Tree Index  

# COMMAND ----------

TREEINDEXDIR = PERSIST_DIR + 'TreeIndex' 

if not os.path.exists(TREEINDEXDIR) or os.listdir(TREEINDEXDIR) == []:
    os.mkdir(TREEINDEXDIR)
    print("Loading Documents...")
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    print("Creating Index...")
    treeindex = GPTTreeIndex.from_documents(documents, service_context=service_context)               ## Tree Index
    print("Persisting Index...")
    treeindex.storage_context.persist(persist_dir=TREEINDEXDIR)
else:
    print("Reading from Index...")
    treeindex = load_index_from_storage(storage_context)

# COMMAND ----------

# MAGIC %md
# MAGIC #### D) Creating (or loading) Keyword Table Indexes (Simple - regex and with embeddings)  
# MAGIC
# MAGIC The GPTKeywordTableIndex implementation extracts the keywords from indexed nodes and uses them to find relevant documents. When we ask a question, first, the implementation will generate keywords from the question. Next, the index searches for the relevant documents and sends them to the LLM.  
# MAGIC The bulk of the work happens at the indexing time. Every node is sent to the LLM to generate keywords. Of course, sending every document to an LLM skyrockets the cost of indexing. Not only because we pay for the tokens but also because calls to the Completion API of OpenAI take longer than their Embeddings API.

# COMMAND ----------

KEYWORDINDEXDIR = PERSIST_DIR + 'KeywordIndex' 

if not os.path.exists(KEYWORDINDEXDIR) or os.listdir(KEYWORDINDEXDIR) == []:
    os.mkdir(KEYWORDINDEXDIR)
    print("Loading Documents...")
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    print("Creating Index...")
    keywordindex = GPTKeywordTableIndex.from_documents(documents, service_context=service_context)        ## Keyword Table Index
    print("Persisting Index...")
    keywordindex.storage_context.persist(persist_dir=KEYWORDINDEXDIR)
else:
    print("Reading from Index...")
    keywordindex = load_index_from_storage(storage_context)

# COMMAND ----------

keywordindex.index_struct

# COMMAND ----------

SIMPLEKEYWORDINDEXDIR = PERSIST_DIR + 'SimpleKeywordIndex' 

if not os.path.exists(SIMPLEKEYWORDINDEXDIR) or os.listdir(SIMPLEKEYWORDINDEXDIR) == []:
    os.mkdir(SIMPLEKEYWORDINDEXDIR)
    print("Loading Documents...")
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    print("Creating Index...")
    simplekeywordindex = GPTSimpleKeywordTableIndex.from_documents(documents, service_context=service_context) ## Simple Keyword Table Index
    print("Persisting Index...")
    simplekeywordindex.storage_context.persist(persist_dir=SIMPLEKEYWORDINDEXDIR)
else:
    print("Reading from Index...")
    simplekeywordindex = load_index_from_storage(storage_context)

# COMMAND ----------

simplekeywordindex.index_struct

# COMMAND ----------

# MAGIC %md
# MAGIC #### E) Creating (or loading) Document Summary Index  
# MAGIC

# COMMAND ----------

DSUMMARYINDEXDIR = PERSIST_DIR + 'DSummaryIndex' 

response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT)

if not os.path.exists(DSUMMARYINDEXDIR) or os.listdir(DSUMMARYINDEXDIR) == []:
    os.mkdir(DSUMMARYINDEXDIR)
    print("Loading Documents...")
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    print("Creating Index...")
    dsummaryindex = GPTDocumentSummaryIndex.from_documents(documents, 
                                                           service_context=service_context,
                                                           response_synthesizer=response_synthesizer)  ## Document Summary Index
    print("Persisting Index...")
    dsummaryindex.storage_context.persist(persist_dir=DSUMMARYINDEXDIR)
else:
    print("Reading from Index...")
    dsummaryindex = load_index_from_storage(storage_context)

# COMMAND ----------

# MAGIC %md
# MAGIC #### F) Creating (or loading) Knowledge Graph Index  
# MAGIC
# MAGIC GPTKnowledgeGraphIndex builds a knowledge graph with keywords and relations between nodes, consuming a lot of resources.  
# MAGIC The default behavior of GPTKnowledgeGraphIndex is based on keywords, but we can use embeddings by specifying the retriever_mode parameter (KGRetrieverMode.EMBEDDING) 

# COMMAND ----------

KGGRAPHINDEXDIR = PERSIST_DIR + 'KGraphIndex' 

if not os.path.exists(KGGRAPHINDEXDIR) or os.listdir(KGGRAPHINDEXDIR) == []:
    os.mkdir(KGGRAPHINDEXDIR)
    print("Loading Documents...")
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    print("Creating Index...")
    kgraphindex = GPTKnowledgeGraphIndex.from_documents(documents, service_context=service_context),  ## Knowledge Graph Index
    print("Persisting Index...")
    kgraphindex.storage_context.persist(persist_dir=KGGRAPHINDEXDIR)
else:
    print("Reading from Index...")
    kgraphindex = load_index_from_storage(storage_context)

# COMMAND ----------

# MAGIC %md ## Retrieving your data  
# MAGIC
# MAGIC ![](https://miro.medium.com/v2/resize:fit:720/format:webp/0*XqAckrehpK4MYt35.jpg)
# MAGIC
# MAGIC LlamaIndex provides a high-level API that facilitates straightforward querying, ideal for common use cases.
# MAGIC ```
# MAGIC query_engine = index.as_query_engine()
# MAGIC response = query_engine.query("your_query")
# MAGIC print(response)
# MAGIC ```
# MAGIC
# MAGIC ```as_query_engine``` builds a default retriever and query engine on top of the index.  
# MAGIC You can check check out the query engine, chat engine and agents sections [here](https://docs.llamaindex.ai/en/stable/module_guides/querying/querying.html)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### A) Retrieving from [List Index](https://docs.llamaindex.ai/en/stable/api_reference/query/retrievers/list.html)  
# MAGIC
# MAGIC LlamaIndex provides embedding support to list indices.  
# MAGIC In addition to each node storing text, each node can optionally store an embedding. During query time, we can use embeddings to do max-similarity retrieval of nodes before calling the LLM to synthesize an answer.  
# MAGIC Since similarity lookup using embeddings (e.g. using cosine similarity) does not require an LLM call, embeddings serve as a cheaper lookup mechanism instead of using LLMs to traverse nodes.

# COMMAND ----------

query_engine = listindex.as_query_engine(similarity_top_k=2,
                                         keyword_filter=["Raiffeisen", "Generative AI"],
                                         response_mode="accumulate",   #https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/response_modes.html
                                         verbose=True,
                                         )
response = query_engine.query("What industries demand upskilling regarding GenAI?")
print(response)

# COMMAND ----------

response = query_engine.query("Will GenAI create new jobs?")
print(response)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### B) Retrieving from [Vector Store Index](https://docs.llamaindex.ai/en/stable/api_reference/query/retrievers/vector_store.html)  
# MAGIC
# MAGIC Check the many [Response Modes](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/response_modes.html)  

# COMMAND ----------

query_engine = vectorstoreindex.as_query_engine(retriever_mode="embedding",
                                                response_mode="compact",
                                                verbose=True)
response = query_engine.query("What industries demand upskilling regarding GenAI?")
print(response)

# COMMAND ----------

response = query_engine.query("Will GenAI create new jobs?")
print(response)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### C) Retrieving from [Tree Index](https://docs.llamaindex.ai/en/stable/api_reference/query/retrievers/tree.html)

# COMMAND ----------


query_engine = treeindex.as_query_engine(response_mode="tree_summarize",  
                                         retriever_mode="all_leaf",  
                                         response_mode="compact",
                                         child_branch_factor=1,
                                         verbose=True
                                        )
response = query_engine.query("What industries demand upskilling regarding GenAI?")
print(response)

# COMMAND ----------

response = query_engine.query("Will GenAI create new jobs?")
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC #### D) Retrieving from [SimpleKeywordTableIndex and KeywordTableIndex](https://docs.llamaindex.ai/en/stable/api_reference/query/retrievers/table.html)

# COMMAND ----------

query_engine = simplekeywordindex.as_query_engine(verbose=True
                                                  response_mode="compact",
                                                  )
response = query_engine.query("What industries demand upskilling regarding GenAI?")
print(response)

# COMMAND ----------

response = query_engine.query("Will GenAI create new jobs?")
print(response)

# COMMAND ----------

query_engine = keywordindex.as_query_engine(verbose=True,
                                            response_mode="compact",
                                        )
response = query_engine.query("What industries demand upskilling regarding GenAI?")
print(response)

# COMMAND ----------

response = query_engine.query("Will GenAI create new jobs?")
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC #### E) Retrieving from [Document Summary Index](https://docs.llamaindex.ai/en/latest/examples/index_structs/doc_summary/DocSummary.htmll)

# COMMAND ----------

query_engine = dsummaryindex.as_query_engine(verbose=True,
                                           response_mode="compact",
                                           )
response = query_engine.query("What industries demand upskilling regarding GenAI?")
print(response)

#response = response_synthesizer.synthesize("query text", nodes=[Node(text="text"), ...])

# COMMAND ----------

dsummaryindex.get_document_summary("What industries demand upskilling regarding GenAI?")

# COMMAND ----------

# MAGIC %md
# MAGIC #### F) Retrieving from [Knowledge Graph Index](https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/KnowledgeGraphDemo.html)

# COMMAND ----------

query_engine = kgraphindex.as_query_engine(verbose=True,
                                           response_mode="compact",
                                           )
response = query_engine.query("What industries demand upskilling regarding GenAI?")
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC GPTKnowledgeGraphIndex comes with an additional benefit. We can retrieve the knowledge graph, and we can even visualize it with the networkx library.

# COMMAND ----------

import networkx as nx
import matplotlib.pyplot as plt
fig = plt.figure(1, figsize=(100, 40), dpi=30)
nx.draw_networkx(kgraphindex.get_networkx_graph(), font_size=18)

# COMMAND ----------

# MAGIC %md
# MAGIC #### F) Pandas Index
# MAGIC
# MAGIC Error: https://stackoverflow.com/questions/77445728/pandasqueryengine-from-llama-index-is-unable-to-execute-code-with-the-following

# COMMAND ----------

df = pd.read_csv("../../Data/bank_data.csv")
index = GPTPandasIndex(df=df) #, service_context=service_context)
query_engine = index.as_query_engine(verbose=True)
response = query_engine.query("What is the size of the dataframe?",)
response

# COMMAND ----------

# MAGIC %md
# MAGIC #### G) SQL Index:
# MAGIC Think about a cool application where you can attach your LLM app to your database and ask questions on top of it.  
# MAGIC This is a non-functional example code  

# COMMAND ----------

# install wikipedia python package
!pip install wikipedia

from llama_index import SimpleDirectoryReader, WikipediaReader
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, select, column

wiki_docs = WikipediaReader().load_data(pages=['Toronto', 'Berlin', 'Tokyo'])

engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()

# create city SQL table
table_name = "city_stats"
city_stats_table = Table(
    table_name,
    metadata_obj,
    Column("city_name", String(16), primary_key=True),
    Column("population", Integer),
    Column("country", String(16), nullable=False),
)
metadata_obj.create_all(engine)

from llama_index import GPTSQLStructStoreIndex, SQLDatabase, ServiceContext
from langchain import OpenAI
from llama_index import LLMPredictor

llm_predictor = LLMPredictor(llm=LLMPredictor(llm=ChatOpenAI(temperature=0, max_tokens=512, model_name='gpt-3.5-turbo')))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

sql_database = SQLDatabase(engine, include_tables=["city_stats"])
sql_database.table_info

# NOTE: the table_name specified here is the table that you
# want to extract into from unstructured documents.
index = GPTSQLStructStoreIndex.from_documents(
    wiki_docs, 
    sql_database=sql_database, 
    table_name="city_stats",
    service_context=service_context
)

# view current table to verify the answer later
stmt = select(
    city_stats_table.c["city_name", "population", "country"]
).select_from(city_stats_table)

with engine.connect() as connection:
    results = connection.execute(stmt).fetchall()
    print(results)

query_engine = index.as_query_engine(
    query_mode="nl"
)
response = query_engine.query("Which city has the highest population?")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Chating with your Documents' Index  
# MAGIC
# MAGIC After examining the many ways in which you can create your indexes, now it's time to 

# COMMAND ----------

# do imports
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent

from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig

query_engine = vectorstoreindex.as_query_engine()
tool_config = IndexToolConfig(query_engine=query_engine, 
                              name=f"Financial Report",
                              description=f"useful for when you want to answer queries about the Apple financial report",
                              tool_kwargs={"return_direct": True}
                              )
toolkit = LlamaToolkit(index_configs=[tool_config])
memory = ConversationBufferMemory(memory_key="chat_history")
llm=ChatOpenAI(temperature=0.2,model_name='gpt-3.5-turbo')
agent_chain = create_llama_chat_agent(
    toolkit,
    llm,
    memory=memory,
    verbose=True
)
while True:
    text_input = input("User: ")
    response = agent_chain.run(input=text_input)
    print(f'Agent: {response}')
