# Databricks notebook source
# MAGIC %md
# MAGIC ## LangChain: Q&A over Documents
# MAGIC
# MAGIC An example might be a tool that would allow you to query a product catalog for items of interest.
# MAGIC
# MAGIC Sources: [Here](https://learn.deeplearning.ai/langchain/lesson/5/question-and-answer),
# MAGIC [here](https://betterprogramming.pub/building-a-multi-document-reader-and-chatbot-with-langchain-and-chatgpt-d1864d47e339) and 
# MAGIC [here](https://python.langchain.com/docs/integrations/vectorstores/faiss)

# COMMAND ----------

#!pip install -q docarray
#!pip install python-docx
!pip install -Uq docx2txt

# COMMAND ----------

# MAGIC %md
# MAGIC ## Installing Packages

# COMMAND ----------

!pip install -q pydantic==1.10.9  #https://stackoverflow.com/questions/76934579/pydanticusererror-if-you-use-root-validator-with-pre-false-the-default-you

# COMMAND ----------

!pip install -q transformers
!pip install -q InstructorEmbedding

# COMMAND ----------

!pip install -q pypdf

# COMMAND ----------

!pip install -q unstructured[pdf]

# COMMAND ----------

#!pip install -q chromadb
!pip install faiss-cpu

# COMMAND ----------

#dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing Packages

# COMMAND ----------

import os
import glob
from pathlib import Path
import pandas as pd
from IPython.display import display, Markdown
#from docx import Document
import tiktoken
#from funcy import lcat, lmap, linvoke

import warnings
warnings.filterwarnings('ignore')

## Langchain LLM Objects
import openai
#from langchain.llms import OpenAI
from langchain.llms import AzureOpenAI
#from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI

## Langchain Prompt Templates
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate

## Langchain Chains 
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain

## Langchain Memory 
from langchain.memory import ConversationBufferMemory

## Langchain Text Splitters
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.text_splitter import TokenTextSplitter
#from langchain.text_splitter import MarkdownHeaderTextSplitter

## Langchain Document Object and Loaders
from langchain.docstore.document import Document
from langchain.schema import Document as LangchainDocument
from langchain.document_loaders import CSVLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

## Langchain Vector Databases
#from langchain.vectorstores import DocArrayInMemorySearch
#from langchain.vectorstores.base import VectorStore
#from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS

## Langchain  Embedding Models
#from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from InstructorEmbedding import INSTRUCTOR

#os.environ['TRANSFORMERS_CACHE'] = "/Workspace/ds-academy-embedded-wave-4/Models/"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading the ´gpt-35-turbo´ model

# COMMAND ----------

openai.api_type = "azure"
openai.api_base = "https://rg-rbi-aa-aitest-dsacademy.openai.azure.com/"
#openai.api_base = "https://chatgpt-summarization.openai.azure.com/"
openai.api_key = os.environ["OPENAI_API_KEY"]

openai_model_name = "gpt-35-turbo"
openai_deploy_name = "model-gpt-35-turbo"
openai.api_version = "2023-07-01-preview"

# COMMAND ----------

llm = AzureChatOpenAI(openai_api_base=openai.api_base,
                      openai_api_version=openai.api_version,
                      deployment_name=openai_deploy_name,
                      openai_api_key=os.environ["OPENAI_API_KEY"],
                      openai_api_type=openai.api_type,
                      temperature=0.9,
                      #max_tokens=4000,
                      )


llm

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading files in the examples folder

# COMMAND ----------

# MAGIC %md
# MAGIC The simplest Q&A chain implementation we can use is the load_qa_chain.  
# MAGIC It loads a chain that allows you to pass in all of the documents you would like to query against using your LLM. 
# MAGIC
# MAGIC ![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*rF3UlC7vWiVFGlXFNZ1XHw.png)
# MAGIC
# MAGIC In this first example, we will use only a PDF document

# COMMAND ----------

fullpath = "/Workspace/ds-academy-embedded-wave-4/ExampleDocs"
docs = os.listdir(fullpath)
docs = [d for d in docs if d.endswith(".pdf")]
for doc in docs:
    print(doc)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating a Document Object

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will instantiate the PDF Loader, load one small document and create a list of Langchain documents object
# MAGIC
# MAGIC Info about the page splitting [here](https://datascience.stackexchange.com/questions/123076/splitting-documents-with-langchain-when-a-sentence-straddles-the-a-page-break)  
# MAGIC You can also define your own document splitter using `pdf_loader.load_and_split()`

# COMMAND ----------

print(f'Loading Document: {fullpath+"/"+docs[3]}')
pdf_loader = PyPDFLoader(fullpath+"/"+docs[3])
documents = pdf_loader.load()
print(f"We have {len(documents)} pages in the pdf file")

print(type(documents))
print(type(documents[0]))

# COMMAND ----------

chain = load_qa_chain(llm=llm, verbose=False)
query = 'What is the document about?'
response = chain.run(input_documents=documents, question=query)
print(response) 

# COMMAND ----------

# MAGIC %md
# MAGIC This method is all good when we only have a short amount of information to send in the [context size of our model](https://platform.openai.com/docs/models/overview).  
# MAGIC However, most LLMs will have a limit on the amount of information that can be sent in a single request. So we will not be able to send all the information in our documents within a single request.  
# MAGIC To overcome this, we need a smart way to send only the information we think will be relevant to our question/prompt.  

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Interacting With a Single PDF Using Embeddings
# MAGIC
# MAGIC We can use embeddings and vector stores to send only relevant information to our prompt.  
# MAGIC The steps we will need to follow are:
# MAGIC
# MAGIC + Split all the documents into small chunks of text
# MAGIC + Pass each chunk of text into an embedding transformer to turn it into an embedding
# MAGIC + Store the embeddings and related pieces of text in a vector store, instead of a list of Langchain document objects
# MAGIC
# MAGIC ![](https://miro.medium.com/v2/resize:fit:828/format:webp/1*FWwgOvUE660a04zoQplS7A.png)
# MAGIC
# MAGIC Let's test with a single, yet bigger PDF file

# COMMAND ----------

print(f'Loading Document: {fullpath+"/"+docs[0]}')
pdf_loader = PyPDFLoader(fullpath+"/"+docs[0])
documents = pdf_loader.load()
print(f"We have {len(documents)} pages in the pdf file")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using Splitters
# MAGIC
# MAGIC Langchain offer different Text Splitters
# MAGIC + RecursiveCharacterTextSplitter: Divides the text into fragments based on characters, starting with the first character. If the fragments turn out to be too large, it moves on to the next character. It offers flexibility by allowing you to define the division characters and fragment size.
# MAGIC + CharacterTextSplitter: Similar to the RecursiveCharacterTextSplitter, but with the ability to define a custom separator for more specific division. By default, it tries to split on characters like “\n\n”, “\n”, “ “, and “”.
# MAGIC + RecursiveTextSplitter: Unlike the previous ones, the RecursiveTextSplitter divides text into fragments based on words or tokens instead of characters. This provides a more semantic view and is ideal for content analysis rather than structure.
# MAGIC + TokenTextSplitter: Uses the OpenAI language model to split text into fragments based on tokens, allowing for precise and contextualized segmentation, ideal for advanced natural language processing applications.
# MAGIC + And some more specific ones
# MAGIC
# MAGIC
# MAGIC We will split the data into chunks of 1,000 characters, with an overlap of 200 characters between the chunks, which helps to give better results and contain the context of the information between chunks

# COMMAND ----------

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                               chunk_overlap=200,
                                               separators=["\n\n", "\n", "\. ", " ", ""],
                                               length_function=len
                                               )
documents = text_splitter.split_documents(documents)
print(len(documents))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Choosing an Embedding Model  
# MAGIC
# MAGIC We could create embeddings with many different transformers. 
# MAGIC We could have used using **OpenAIEmbeddings**, but then we would have to pay for each token sent to the API. In our case, we will create our vectorDB using **InstructEmbeddings** transformer from **[Hugging Face](https://huggingface.co/hkunlp/instructor-xl)** to provide embeddings from our text chunks.  

# COMMAND ----------

#openai_embeddings = OpenAIEmbeddings(deployment="model-text-embedding-ada-002", chunk_size = 1)

# COMMAND ----------

'''
from InstructorEmbedding import INSTRUCTOR

modelpath = "/Workspace/ds-academy-embedded-wave-4/Models/model.bin"
try:
    instruct_embeddings = INSTRUCTOR(modelpath)
    print("Successfully loaded Model locally")
except:
    print("Loading from the Web")
    instruct_embeddings = INSTRUCTOR('hkunlp/instructor-xl')
    #instruct_embeddings.save(modelpath)
    #print(f"Saved model to {modelpath}")
'''

instruct_embeddings = HuggingFaceInstructEmbeddings(query_instruction="Represent the query for retrieval: ", model_name="hkunlp/instructor-xl") 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vector Databases
# MAGIC
# MAGIC ![Vector Databases](https://miro.medium.com/v2/resize:fit:828/format:webp/1*vIkxM-u3zrkHMZuIRURc0A.png)
# MAGIC
# MAGIC There are [many Vector Databases](https://thenewstack.io/top-5-vector-database-solutions-for-your-ai-project/)  products, both paid and open source, that could be used. 
# MAGIC We have first tried [ChromaDB](https://www.trychroma.com/), but some incompatibilities with the current versions of Python motivated us to try [FAISS](https://faiss.ai/) (from Meta)
# MAGIC
# MAGIC We set all the db information to be stored inside the `/Workspace/ds-academy-embedded-wave-4/VectorDB`, so it doesn't clutter up our source files.  

# COMMAND ----------

# MAGIC %md
# MAGIC First attempt with ChromaDb (commented)

# COMMAND ----------

#vectordb = Chroma.from_documents(documents,
#                                 #embedding=openai_embeddings,
#                                 embedding=instruct_embeddings,
#                                 persist_directory='/Workspace/ds-academy-embedded-wave-4/VectorDB'
#)
#vectordb.persist()

# COMMAND ----------

# MAGIC %md
# MAGIC Deleting previous databases from the folder we have create to store the files (only if creating new)

# COMMAND ----------

files = glob.glob('/Workspace/ds-academy-embedded-wave-4/VectorDB/*')
for f in files:
    os.remove(f)

# COMMAND ----------

# MAGIC %md
# MAGIC Loading all PDF documents into the Vector Database

# COMMAND ----------

vectordb = FAISS.from_documents(documents, 
                                embedding=instruct_embeddings,
                               )
#print(f"There are {vectordb.ntotal} documents in the index")
vectordb.save_local('/Workspace/ds-academy-embedded-wave-4/VectorDB/')

# COMMAND ----------

# MAGIC %md
# MAGIC Once we have loaded our content as embeddings into the vector store, we are back to a similar situation as to when we only had one PDF to interact with. As in, we are now ready to pass information into the LLM prompt.  
# MAGIC However, instead of passing in all the documents as a source for our context to the chain, as we did initially, we will pass in our vector store as a source/retriever, and the chain will retrieve only the relevant text based on our question and send that information only inside the LLM prompt.
# MAGIC
# MAGIC ![](https://miro.medium.com/v2/resize:fit:828/format:webp/1*leoW-Pn0ohWalrUBbzdidA.png)
# MAGIC
# MAGIC First we will only use the RetrievalQA chain, which will use our vector store as a source for the context information.
# MAGIC
# MAGIC Again, the chain will wrap our prompt with some text, instructing it to only use the information provided for answering the questions.  
# MAGIC So the prompt we end up sending to the LLM something that looks like this:
# MAGIC
# MAGIC     Use the following pieces of context to answer the question at the end.
# MAGIC     If you don't know the answer, just say that you don't know, don't try to
# MAGIC     make up an answer.
# MAGIC
# MAGIC     {context} // i.e the chunks of text retrieved deemed to be most semantically
# MAGIC               // relevant to our question
# MAGIC
# MAGIC     Question: {query} // i.e our actualy query
# MAGIC     Helpful Answer:

# COMMAND ----------

# MAGIC %md
# MAGIC Loading the recently created Vector Database object

# COMMAND ----------

docsearch = FAISS.load_local("/Workspace/ds-academy-embedded-wave-4/VectorDB/", instruct_embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we could query our documents directly using the native similarity search from the vector DB:

# COMMAND ----------

query = "What are the documents about?"
result = docsearch.similarity_search(query)
print(result[0].page_content)

# COMMAND ----------

# MAGIC %md
# MAGIC We can also search using a score function and a maximum number of documents in return

# COMMAND ----------

query = "What are the documents about?"
result = docsearch.similarity_search_with_score(query, k=2)
for r in result:
    print(r)
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC But it is much better to query the document with the Q&A chain.  
# MAGIC Now we create a Retrieval chain using the Vector Database object:

# COMMAND ----------

qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       retriever=docsearch.as_retriever(),
                                       #retriever=docsearch.as_retriever(search_kwargs={'k': 7}),
                                       return_source_documents=True)

# COMMAND ----------

query = "What is Data lakehouse?"
result = qa_chain(query)
print(result['result'])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Adding Chat History
# MAGIC Now, if we want to take things one step further, we can also make it so that our chatbot will remember any previous questions.
# MAGIC
# MAGIC Implementation-wise, all that happens is that on each interaction with the chatbot, all of our previous conversation history, including the questions and answers, needs to be passed into the prompt. That is because the LLM does not have a way to store information about our previous requests, so we must pass in all the information on every call to the LLM.
# MAGIC
# MAGIC Fortunately, LangChain also has a set of classes that let us do this out of the box. This is called the ConversationalRetrievalChain, which allows us to pass in an extra parameter called chat_history , which contains a list of our previous conversations with the LLM.

# COMMAND ----------

qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                 retriever=docsearch.as_retriever(),
                                                 return_source_documents=True)

# COMMAND ----------

# MAGIC %md
# MAGIC The chain run command accepts the chat_history as a parameter. We must manually build up this list based on our conversation with the LLM.  
# MAGIC The chain does not do this out of the box, so for each question and answer, we will build up a list called chat_history , which we will pass back into the chain run command each time.

# COMMAND ----------

chat_history = []
while True:
    # this prints to the terminal, and waits to accept an input from the user
    query = input('Prompt: ')
    # give us a way to exit the script
    if query == "exit" or query == "quit" or query == "q":
        print('Exiting')
        break
    # we pass in the query to the LLM, and print out the response. As well as
    # our query, the context of semantically relevant information from our
    # vector store will be passed in, as well as list of our chat history
    result = qa_chain({'question': query, 'chat_history': chat_history})
    print('Answer: ' + result['answer'])
    # we build up the chat_history list, based on our question and response
    # from the LLM, and the script then returns to the start of the loop
    # and is again ready to accept user input.
    chat_history.append((query, result['answer']))

# COMMAND ----------

chat_history

# COMMAND ----------

# MAGIC %md
# MAGIC #### Interacting With Multiple Document types  
# MAGIC If you remember, the Documents created from our PDF Document Loader is just a list of parts of one Documents. So to increase our base of documents to interact with, we can just add more Documents to this list.
# MAGIC
# MAGIC Now we can simply iterate over all of the files in that folder, and convert the information in them into Documents. From then onwards, the process is the same as before. We just pass our list of documents to the text splitter, which passes the chunked information to the embeddings transformer and vector store.
# MAGIC
# MAGIC So, in our case, we want to be able to handle pdfs, Microsoft Word documents, and text files. We will iterate over the docs folder, handle files based on their extensions, use the appropriate loaders for them, and add them to the documentslist, which we then pass on to the text splitter.

# COMMAND ----------

# MAGIC %md
# MAGIC First we are going to delete the old VectorDB

# COMMAND ----------

files = glob.glob('/Workspace/ds-academy-embedded-wave-4/VectorDB/*')
for f in files:
    os.remove(f)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's now create Langchain Document objects for all different files in our storage folder

# COMMAND ----------

fullpath = "/Workspace/ds-academy-embedded-wave-4/ExampleDocs/"
documents = []
for filename in os.listdir(fullpath):
    print(f"Ingesting document {filename}")
    if filename.endswith('.pdf'):
        pdf_path = fullpath + filename
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif filename.endswith('.docx') or filename.endswith('.doc'):
        doc_path = fullpath + filename
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif filename.endswith('.txt'):
        text_path = fullpath + filename
        loader = TextLoader(text_path)
        documents.extend(loader.load())

# COMMAND ----------

# MAGIC %md
# MAGIC Checking How many objects were created:

# COMMAND ----------

print(len(documents))
for d in documents[0:10]:
    print(d.metadata)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we are going to split the texts as we have done before: 

# COMMAND ----------

#text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"])

chunked_documents = text_splitter.split_documents(documents)

# COMMAND ----------

print(len(chunked_documents))
for d in chunked_documents[0:5]:
    print(d.metadata)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we are going to add documents to the previously created Vector Database index.

# COMMAND ----------

print(f"We have {len(vectordb.docstore._dict)} documents in the collection")
vectordb.add_documents(chunked_documents,
                       embedding=instruct_embeddings,
                       )
print(f"We have {len(vectordb.docstore._dict)} documents in the collection")
vectordb.save_local('/Workspace/ds-academy-embedded-wave-4/VectorDB/')

# COMMAND ----------

# MAGIC %md
# MAGIC The vector database does not distinguish which documents were indexed before, so we have to take care when ingesting to avoid duplicates

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can chat with our documents from multiple types via LLM 

# COMMAND ----------

pdf_qa = ConversationalRetrievalChain.from_llm(llm,
                                               retriever=vectordb.as_retriever(),
                                               return_source_documents=True,
                                               verbose=False
                                               )

chat_history = []
print(f"---------------------------------------------------------------------------------")
print('Welcome to the DocBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')
while True:
    query = input(f"Prompt: ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        break
    if query == '':
        continue
    result = pdf_qa({"question": query, "chat_history": chat_history})
    print(f"Answer: " + result["answer"])
    chat_history.append((query, result["answer"]))

# COMMAND ----------

chat_history

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bonus: operations among Vector Databases

# COMMAND ----------

# ------------------------------
# all the function definitions
# ------------------------------
import os
import openai
import pypdf
#import pandas as pd

#import json
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.llms import HuggingFacePipeline
#from PyPDF2 import PdfReader
from functools import partial
#from funcy import lmap
from typing import Tuple, Callable
from typing import Any
from io import StringIO
from langchain.text_splitter import CharacterTextSplitter
from huggingface_hub import login
from PIL import Image

import torch
import transformers

import streamlit as st

# # --------------------------------------------------------------
# # Providing the access token to Azure OpenAI
# # This only works if you have access to the respective use cases
# # --------------------------------------------------------------
#os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope="llm-usecases", key="AZURE_TOKEN")
os.environ["OPENAI_API_KEY"] = "a6a3baa768694a448134f530041952c6"
os.environ["OPENAI_API_VERSION"] = "2022-12-01"

openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = "https://rg-rbi-aa-aitest-semantic-vectordb.openai.azure.com/"

openAI_text_llm = AzureOpenAI(deployment_name="model-text-davinci-003", temperature=0)

# -----------------------------------------------------------------
# Load OpenAI Chat model
# -----------------------------------------------------------------
version = "2023-07-01-preview"

#os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope="llm-usecases", key="AZURE_TOKEN")
os.environ["OPENAI_API_KEY"] = "a6a3baa768694a448134f530041952c6" # should be hidden in streamlit
os.environ["OPENAI_API_VERSION"] = version
os.environ["OPENAI_API_BASE"] = "https://rg-rbi-aa-aitest-semantic-vectordb.openai.azure.com/"

openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_type = "azure"
openai.api_version = version
openai.api_base = "https://rg-rbi-aa-aitest-semantic-vectordb.openai.azure.com/"

openAI_chat_llm = AzureChatOpenAI(deployment_name="model-gpt-35-turbo", temperature=0)

def load_embeddings():
    # Load your embeddings here
    # This function will be cached and only executed once
    embeddings = HuggingFaceInstructEmbeddings(
                        query_instruction="Represent the query for retrieval: ", 
                        model_name="hkunlp/instructor-xl"
                        )
    return embeddings

def pdf_to_pages(file):
	"extract text (pages) from pdf file"
	pages = []
	pdf = pypdf.PdfReader(file)
	for p in range(len(pdf.pages)):
		page = pdf.pages[p]
		text = page.extract_text()
		pages += [text]
	return pages

def doc_chain(file):

    pdfpages = pdf_to_pages(file)
    
    documents_content = '\n'.join(page for page in pdfpages)

    # Textsplitter
    text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 500,
            chunk_overlap  = 200,
            length_function = len
        )

    doc_chunks = text_splitter.split_text(documents_content)

    # Call the function to load the embeddings
    instruct_embeddings = load_embeddings()

    vector_db = Chroma.from_texts(doc_chunks, instruct_embeddings)
    chain = load_qa_chain(openAI_text_llm, chain_type="stuff")  
    
    return chain, vector_db

def ask(question:str):
    retriever = vector_db.as_retriever()
    docs = retriever.get_relevant_documents(question)
    answer = chain.run(input_documents=docs, question=question).strip()
    return answer

st.title("APEX Chatbot")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# no file has been loaded
if "file" not in st.session_state:
    st.session_state.file = ""

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:

    image = Image.open('/Workspace/Repos/david.eschwe@rbinternational.com/LLM_Sandbox/Docs/apex_logo.jpg')

    st.image(image)

    uploaded_file = st.file_uploader("Choose a file")

    if (uploaded_file is not None) and st.session_state.file == "":

        st.session_state.file = uploaded_file.name

        chain, vector_db = doc_chain(uploaded_file)
    
    # new file has been uploaded
    if (uploaded_file is not None) and st.session_state.file != uploaded_file.name:

        st.session_state.file = uploaded_file.name

        chain, vector_db = doc_chain(uploaded_file)

    # file deleted
    if (uploaded_file is None):

        st.session_state.file = ""

# React to user input
if prompt := st.chat_input("What is up?"):
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.file != "":
        
        answer_from_doc = ""

        chain, vector_db = doc_chain(uploaded_file)
        answer_from_doc = ask(prompt)

        # SystemPrompt = "Please answer the following question as diligently as possible. Answer with 'I do not know' in case you do not know the answer."

        # response = openAI_chat_llm([
        #         SystemMessage(content=SystemPrompt),
        #         HumanMessage(content=prompt)
        #         ])

        # sum_prompt = "Summarize the two answers to the following prompt: " + prompt + "\n\ (1): " + response.content + "\n\ (2): " + answer_from_doc

        # SystemPrompt = "Create a concise and consistent summary that aggregates all inputs."

        # sum_response = openAI_chat_llm([
        #         SystemMessage(content=SystemPrompt),
        #         HumanMessage(content=sum_prompt)
        #         ])

        # # Display assistant response in chat message container
        # with st.chat_message("assistant"):
        #     st.markdown(f"APEX: {sum_response.content}")
        # # Add assistant response to chat history
        # st.session_state.messages.append({"role": "assistant", "content": sum_response.content})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(f"APEX: {answer_from_doc}")
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer_from_doc})
        
    else:
        
        SystemPrompt = "Please answer the following question as diligently as possible. Answer with 'I do not know' in case you do not know the answer."

        response = openAI_chat_llm([
                    SystemMessage(content=SystemPrompt),
                    HumanMessage(content=prompt)
                    ])

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            
            st.markdown(f"APEX: {response.content}")

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response.content})
        
        

