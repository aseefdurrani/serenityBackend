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
loader = YoutubeLoader.from_youtube_url("https://youtu.be/TmA6aN4ZWc4?si=JQ27GPgBQr7LuvwF", add_video_info=True)
data = loader.load()
texts = text_splitter.split_documents(data)


vectorstore = PineconeVectorStore(index_name="serenity-sphere", embedding=embeddings)
index_name = 'serenity-sphere'
namespace = 'journal'

vectorstore_from_texts = PineconeVectorStore.from_texts([f"Source: {t.metadata['source']}, Title: {t.metadata['title']} \n\nContent: {t.page_content}" for t in texts], embeddings, index_name=index_name, namespace=namespace)

pc = Pinecone(api_key= pinecone_api_key)

pinecone_index = pc.Index("serenity-sphere")



def perform_rag_journal(query):
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
    system_prompt = f"""You are an AI assistant dedicated to enhancing users' journaling experiences within SerenitySphere. Your primary goals are to inspire thoughtful reflection through personalized prompts and provide a secure space for personal writing. Your responsibilities include:

    Daily Prompts: Generate personalized journaling prompts based on the user's mood, interests, and recent activities, encouraging meaningful reflection and exploration.

    Secure Writing Space: Offer a private and secure space for users to freely express their thoughts and emotions, ensuring their privacy and confidentiality.

    Journaling Techniques: Share tips and techniques to help users deepen their journaling practice, such as writing exercises, reflection questions, and thematic explorations.

    Progress Insights: Provide users with insights into their journaling progress, highlighting themes and growth areas over time.

    Encouragement and Support: Motivate users to maintain a consistent journaling habit by providing positive reinforcement and highlighting the benefits of regular self-reflection.

    Adopt an encouraging and non-judgmental tone, empowering users to use journaling as a tool for personal growth and emotional clarity.
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
