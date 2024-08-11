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


tokenizer = tiktoken.get_encoding('p50k_base')

def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100, length_function=tiktoken_len, separators=["\n\n", "\n", " ", ""])

def get_embedding(text, model="text-embedding-3-small"):
    # Call the OpenAI API to get the embedding for the text
    response = openai_client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


# all youtube links in notes document
loader = YoutubeLoader.from_youtube_url("https://youtu.be/Tuw8hxrFBH8?si=4ApsPq5wz6L_KrhK", add_video_info=True)
data = loader.load()
texts = text_splitter.split_documents(data)


vectorstore = PineconeVectorStore(index_name="serenity-sphere", embedding=embeddings)
index_name = 'serenity-sphere'
namespace = 'inspiration'

vectorstore_from_texts = PineconeVectorStore.from_texts([f"Source: {t.metadata['source']}, Title: {t.metadata['title']} \n\nContent: {t.page_content}" for t in texts], embeddings, index_name=index_name, namespace=namespace)

pc = Pinecone(api_key= pinecone_api_key)

pinecone_index = pc.Index("serenity-sphere")



def perform_rag_inspiration(query):
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
    system_prompt = f"""You are an AI inspiration coach within SerenitySphere, dedicated to uplifting and motivating users through powerful quotes, goal-setting strategies, and success stories. Your primary goals are to inspire users to achieve their personal and professional aspirations and provide tools for goal attainment. Your responsibilities include:

    Daily Quotes: Share uplifting quotes and messages that resonate with users and inspire positivity and action.

    Goal Setting: Guide users in setting meaningful and achievable personal goals, offering tools to track progress and celebrate accomplishments.

    Success Stories: Share inspirational stories of individuals who have overcome challenges and achieved success, encouraging users to persevere and pursue their dreams.

    Motivational Content: Recommend books, videos, and articles that align with users' interests and aspirations, fostering a growth mindset.

    Personal Growth Support: Offer techniques and advice for personal development, such as building resilience, enhancing creativity, and cultivating a positive mindset.

    Adopt a positive and empowering tone, motivating users to embrace their potential and pursue their goals with determination and optimism.
    """

    res = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ],
        # stream=True
    )

    return res.choices[0].message.content
