# Databricks notebook source
# MAGIC %md
# MAGIC ## Ask Questions to multiple documents

# COMMAND ----------

# MAGIC %pip install -qU funcy
# MAGIC %pip install -qU huggingface_hub
# MAGIC %pip install -qU InstructorEmbedding
# MAGIC %pip install -qU langchain
# MAGIC %pip install -qU chromadb
# MAGIC %pip install -qU openpyxl
# MAGIC %pip install -Uq docx2txt
# MAGIC %pip install -qU python-docx
# MAGIC %pip install -qU sentence-transformers
# MAGIC %pip install -qU tiktoken
# MAGIC %pip install -qU torch
# MAGIC %pip install -qU pypdf
# MAGIC %pip install -qU xformers
# MAGIC %pip install -qU langchainhub
# MAGIC %pip install -qU llama-cpp-python
# MAGIC %pip install -qU accelerate
# MAGIC %pip install -qU panel
# MAGIC %pip install -qU streamlit

# COMMAND ----------

!pip show langchain

# COMMAND ----------

# MAGIC %md
# MAGIC ### Importing Packages

# COMMAND ----------

# all the function definitions
import os
import pandas as pd
import json

from functools import partial
from funcy import lmap
from typing import Tuple, Callable
from typing import Any

import torch
import transformers

import openai
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
#from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.docstore.document import Document
from langchain.schema import Document as LangchainDocument
from langchain.document_loaders import CSVLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logging to Hugging Face

# COMMAND ----------

from getpass import getpass
from huggingface_hub import login

login(token=getpass("Huggingface Token:"))

# COMMAND ----------

#model = "meta-llama/Llama-2-13b-chat-hf"
#model = "meta-llama/Llama-2-13b-hf"
model = "meta-llama/Llama-2-7b-chat-hf"
#model = "meta-llama/Llama-2-7b-hf"

tokenizer = transformers.AutoTokenizer.from_pretrained(model)

llama_chat = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    temperature=0.05,
    max_new_tokens=1000,
    #trust_remote_code=True
)

llm = HuggingFacePipeline(pipeline=llama_chat)

# COMMAND ----------

# -----------------------------------------------------
# Load InstructXL embeddings used for OpenAI GPT models
# -----------------------------------------------------
instruct_embeddings = HuggingFaceInstructEmbeddings(
    query_instruction="Represent the query for retrieval: ", 
    model_name="hkunlp/instructor-xl"
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Creating a retrieval pipeline  
# MAGIC
# MAGIC We can use embeddings and vector stores to send only relevant information to our prompt.  
# MAGIC The steps we will need to follow are:
# MAGIC
# MAGIC + Split all the documents into small chunks of text
# MAGIC + Pass each chunk of text into an embedding transformer to turn it into an embedding
# MAGIC + Store the embeddings and related pieces of text in a vector store, instead of a list of Langchain document objects
# MAGIC
# MAGIC ![](https://miro.medium.com/v2/resize:fit:828/format:webp/1*FWwgOvUE660a04zoQplS7A.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setting up Docs and Vector Database folders

# COMMAND ----------

pathdocs = "/Workspace/ds-academy-embedded-wave-4/ExampleDocs/"
docs = os.listdir(pathdocs)
docs = [d for d in docs] # if d.endswith(".pdf")]
for doc in docs:
    print(doc)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating the Document Objects
# MAGIC
# MAGIC Now we will instantiate the PDF Loader, load one small document and create a list of Langchain documents object  
# MAGIC Info about the page splitting [here](https://datascience.stackexchange.com/questions/123076/splitting-documents-with-langchain-when-a-sentence-straddles-the-a-page-break)  
# MAGIC You can also define your own document splitter using `pdf_loader.load_and_split()`

# COMMAND ----------

documents = []
for filename in os.listdir(pathdocs):
    print(f"Ingesting document {filename}")
    if filename.endswith('.pdf'):
        pdf_path = pathdocs + filename
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif filename.endswith('.docx') or filename.endswith('.doc'):
        doc_path = pathdocs + filename
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif filename.endswith('.txt'):
        text_path = pathdocs + filename
        loader = TextLoader(text_path)
        documents.extend(loader.load())

# COMMAND ----------

chunk_size = 1000
chunk_overlap = 200
#text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=[" ", ",", "\n"])

chunked_documents = text_splitter.split_documents(documents)

# COMMAND ----------

print(len(chunked_documents))
for d in chunked_documents[0:5]:
    print(d.metadata)

# COMMAND ----------

# MAGIC %md
# MAGIC ### A quick example on similarity search using Cosine Distance
# MAGIC
# MAGIC Naive [Implementation of Cosine Similarity Search](https://github.com/chroma-core/chroma/blob/main/chromadb/utils/distance_functions.py)

# COMMAND ----------

def cosine_similarity (vector1: list, vector2: list):
    if len(vector1) != len(vector2):
        return None
    else:
        scalar_product = 0
        norm1 = 0
        norm2 = 0
        NORM_EPS = 1e-30
        for i in range(0, len(vector1)):
            scalar_product += vector1[i]*vector2[i]
            norm1 += vector1[i]*vector1[i] 
            norm2 += vector2[i]*vector2[i]
        return 1 - (scalar_product / ((norm1**0.5 + NORM_EPS) * (norm2**0.5 + NORM_EPS)))

# COMMAND ----------

text1 = "Hello World"
text2 = "Hello"

a = instruct_embeddings.embed_query(text1)
b = instruct_embeddings.embed_query(text2)

# 768 dimensions for the embeddings
len(a)

# COMMAND ----------

print(cosine_similarity(a,b))

vectordb_text = Chroma.from_texts(texts=[text1], embedding=instruct_embeddings)
response = vectordb_text.similarity_search(text2, 1)
#response = vectordb_text.similarity_search_by_vector(b)

print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating our Vector Database  
# MAGIC We are using ChromaDB in this notebook

# COMMAND ----------

persist_directory = '/Workspace/ds-academy-embedded-wave-4/VectorDB2/'
vectordb = Chroma.from_documents(documents=chunked_documents, 
                                 #embedding=instruct_embeddings, 
                                 embedding_function=instruct_embeddings, 
                                 persist_directory=persist_directory)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Retrieving Documents using Similarity Search

# COMMAND ----------

retriever = vectordb.as_retriever(search_kwargs={"k": 8})
#retriever = vectordb.as_retriever(search_kwargs={"k": 1, "filter": {"page":10}})

print(retriever.search_kwargs)
print(retriever.search_type)

# COMMAND ----------

docs = retriever.get_relevant_documents("What is Delta ?")
for d in docs:
    print(d, "\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using an LLM to improve retrieval

# COMMAND ----------

qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

# COMMAND ----------

query = "Who is Romeo?"
llm_response = qa_chain(query)

# COMMAND ----------

#print(llm_response)
print(llm_response["query"])
print(llm_response["result"])
#print(llm_response["source_documents"][0].metadata["page"])
print(llm_response["source_documents"][0].metadata["source"])
print(llm_response["source_documents"][0].page_content)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using a Langchain Chain 

# COMMAND ----------

from langchain import PromptTemplate

template_string = """
<<SYS>>
You are a helpful, respectful and honest assistant. 
You are working in a european bank in the area of Data Science Modeling. 
You shall answer questions on the topic.

Always help the user finding the best answers from the provided documentation. 

If you are unsure about an answer, truthfully say "I don't know"
<</SYS>>

[INST] 
Remember you are an assistant  

User: {question}
[/INST]
"""
prompt_template = PromptTemplate.from_template(template_string)

# COMMAND ----------

query = "Why any analytical model will degrade over time, according to the AA Models Monitoring Framework?"
#query = "What are the most recent advances in Natural Language Processing?"

llm_response = qa_chain(prompt_template.format(question=query))
print(llm_response['result'])
