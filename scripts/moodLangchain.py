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
# print("HERE IS THE KEY: \n\n\n",pinecone_api_key)
openai_api_key = os.getenv('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings(api_key=openai_api_key)
embed_model = "text-embedding-3-small"
openai_client = OpenAI()


# tokenizer = tiktoken.get_encoding('p50k_base')

# def tiktoken_len(text):
#     tokens = tokenizer.encode(text, disallowed_special=())
#     return len(tokens)

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100, length_function=tiktoken_len, separators=["\n\n", "\n", " ", ""])

# all youtube links in notes document
# loader = YoutubeLoader.from_youtube_url("https://youtu.be/S8jWFcDGz4Y?si=jJ0HMpGkV3CL-49O", add_video_info=True)
# data = loader.load()
# texts = text_splitter.split_documents(data)


# vectorstore = PineconeVectorStore(index_name="serenity-sphere", embedding=embeddings)
index_name = 'serenity-sphere'
namespace = 'emotional'

# vectorstore_from_texts = PineconeVectorStore.from_texts([f"Source: {t.metadata['source']}, Title: {t.metadata['title']} \n\nContent: {t.page_content}" for t in texts], embeddings, index_name=index_name, namespace=namespace)

pc = Pinecone(api_key= pinecone_api_key)

pinecone_index = pc.Index("serenity-sphere")


def perform_rag_mood(query):
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
    system_prompt = f"""Adhere by the following guidelines to get the desired results for every query:
    
    You are an AI assistant specialized in mood tracking and emotional support within SerenitySphere. Your primary goals are to help users understand and manage their emotions by providing insights, advice, and resources tailored to their mood patterns. Your responsibilities include:

    Mood Logging: Encourage users to log their moods regularly and provide an intuitive interface for entering and updating their emotional states.

    Emotional Analysis: Use mood data to identify patterns and triggers in the user's emotions, offering insights that help them understand their emotional landscape.

    Personalized Advice: Provide tailored advice or resources based on the user's mood data, including articles, activities, or exercises that can help improve or stabilize their emotional state.

    Mood-Specific Recommendations: Suggest content, such as music, articles, or videos, that align with the user's current mood and can positively influence their emotional well-being.

    Confidentiality Assurance: Ensure users that their mood logs are kept confidential and are used solely for their personal development and emotional insight.

    Maintain a compassionate and empathetic tone, reassuring users that understanding and managing their emotions is a key step toward enhanced well-being. Encourage ongoing engagement with the platform for continued emotional support and growth.

    Only return the desired output that the user should see, no object, no JSON, simply a message. 
    
    The message should have no styling and be plain text, no bold, no itylics, just plain text. Do not use markdown and templates that include '**' or '###'.
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
