# Databricks notebook source
# MAGIC %md
# MAGIC ### RAG Pipeline with LlamaIndex  
# MAGIC Sources: [1](https://lmy.medium.com/comparing-langchain-and-llamaindex-with-4-tasks-2970140edf33), [2](https://docs.llamaindex.ai/en/stable/), [3](https://github.com/run-llama/llama_index)
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

# COMMAND ----------

!pip install -qU openai
!pip install -qU llama-index
dbutils.library.restartPython()

# COMMAND ----------

import os
import glob
from pathlib import Path
from IPython.display import display, Markdown
import pandas as pd
import tiktoken
#from funcy import lcat, lmap, linvoke
import warnings
warnings.filterwarnings('ignore')

import openai
from openai import OpenAI
from openai import AzureOpenAI

openai.api_type = "azure"
azure_endpoint = "https://rg-rbi-aa-aitest-dsacademy.openai.azure.com/"
#azure_endpoint = "https://chatgpt-summarization.openai.azure.com/"

openai.api_version = "2023-07-01-preview"
openai.api_key = os.environ["OPENAI_API_KEY"]
deployment_name = "model-gpt-35-turbo"
openai_model_name = "gpt-35-turbo"

client = AzureOpenAI(api_key=openai.api_key,
                     api_version=openai.api_version,
                     azure_endpoint=azure_endpoint,
                     )

def ask(prompt):
    try:
        chat_completion = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{prompt}"},
            ],
            )
        return(chat_completion.choices[0].message.content)
    except openai.error.APIError as e:
        print(f"OpenAI API returned an API Error: {e}")
    except openai.error.AuthenticationError as e:
        print(f"OpenAI API returned an Authentication Error: {e}")
    except openai.error.APIConnectionError as e:
        print(f"Failed to connect to OpenAI API: {e}")
    except openai.error.InvalidRequestError as e:
        print(f"Invalid Request Error: {e}")
    except openai.error.RateLimitError as e:
        print(f"OpenAI API request exceeded rate limit: {e}")
    except openai.error.ServiceUnavailableError as e:
        print(f"Service Unavailable: {e}")
    except openai.error.Timeout as e:
        print(f"Request timed out: {e}")
    except:
        print("An exception has occured.")
