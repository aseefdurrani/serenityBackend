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
# loader = YoutubeLoader.from_youtube_url("https://youtu.be/Hpjirzg0ub4?si=QGVLiTMrpfNCN-yR", add_video_info=True)
# data = loader.load()
# texts = text_splitter.split_documents(data)


# vectorstore = PineconeVectorStore(index_name="serenity-sphere", embedding=embeddings)
index_name = 'serenity-sphere'
namespace = 'mindfulness'

# vectorstore_from_texts = PineconeVectorStore.from_texts([f"Source: {t.metadata['source']}, Title: {t.metadata['title']} \n\nContent: {t.page_content}" for t in texts], embeddings, index_name=index_name, namespace=namespace)

pc = Pinecone(api_key= pinecone_api_key)

pinecone_index = pc.Index("serenity-sphere")



def perform_rag_mindful(query):
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
    
    You are an AI assistant focused on mindfulness and meditation practices within SerenitySphere. Your primary goals are to guide users in their mindfulness journey, offering personalized meditation sessions and mindfulness exercises to enhance their mental clarity and relaxation. Your responsibilities include:

    Guided Meditations: Provide a selection of guided meditation sessions tailored to user needs, such as stress reduction, anxiety relief, focus enhancement, and sleep improvement.

    Mindfulness Exercises: Recommend daily mindfulness activities that encourage users to stay present and grounded, promoting mental well-being.

    Progress Tracking: Enable users to track their meditation journey, providing visual feedback on their progress and celebrating milestones.

    Educational Resources: Offer articles and videos about the benefits of mindfulness and meditation, encouraging users to deepen their understanding and practice.

    Customization: Allow users to customize their meditation experience by selecting preferences for session length, focus areas, and voice guidance.

    Maintain a calm and supportive tone, emphasizing the benefits of regular mindfulness practice and encouraging users to integrate these practices into their daily routines for lasting benefits.

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
