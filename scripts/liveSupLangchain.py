from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, WebBaseLoader, YoutubeLoader, DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
#from google.colab import userdata
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import numpy as np
import tiktoken
import os

load_dotenv()

pinecone_api_key = os.getenv('PINECONE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings(api_key=openai_api_key)
embed_model = "text-embedding-3-small"
openai_client = OpenAI()

index_name = 'serenity-sphere'
namespace = 'journal'


pc = Pinecone(api_key= pinecone_api_key)

pinecone_index = pc.Index("serenity-sphere")



def perform_rag_support(query):
    raw_query_embedding = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )

    query_embedding = raw_query_embedding.data[0].embedding


    top_matches = pinecone_index.query(vector=query_embedding, top_k=10, include_metadata=True, namespace=namespace)


    # Get the list of retrieved texts
    contexts = [item['metadata']['text'] for item in top_matches['matches']]

    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    # Modify the prompt below as need to improve the response quality
    system_prompt = f"""
    The only message you should output is: "This feature will be released soon, please checkout our other channels!" And things along this line.

    Only return the desired output that the user should see, no object, no JSON, simply a message. 
    
    The message should have no styling and be plain text, no bold, no itylics, just plain text. Do not use markdown and templates that include '**' or '###'.

    Keep a human touch in the message, make it sound like a human wrote it, not a robot.
    """

    res = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ],
        stream=True, # enable streaming to get intermediate results
    )

    for chunk in res:
        content = chunk.choices[0].delta.content
        if content:
            yield content
