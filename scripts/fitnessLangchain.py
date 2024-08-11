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

# def cosine_similarity_between_words(sentence1, sentence2):
#     # Get embeddings for both words
#     embedding1 = np.array(get_embedding(sentence1))
#     embedding2 = np.array(get_embedding(sentence2))

#     # Reshape embeddings for cosine_similarity function
#     embedding1 = embedding1.reshape(1, -1)
#     embedding2 = embedding2.reshape(1, -1)

#     # print("Embedding for Sentence 1:", embedding1)
#     # print("\nEmbedding for Sentence 2:", embedding2)

#     # Calculate cosine similarity
#     similarity = cosine_similarity(embedding1, embedding2)
#     return similarity[0][0]

# all youtube links in notes document
loader = YoutubeLoader.from_youtube_url("https://youtu.be/2tM1LFFxeKg?si=zBmjd_yuoeEwJpti", add_video_info=True)
data = loader.load()
texts = text_splitter.split_documents(data)


vectorstore = PineconeVectorStore(index_name="serenity-sphere", embedding=embeddings)
index_name = 'serenity-sphere'
namespace = 'fitness'

vectorstore_from_texts = PineconeVectorStore.from_texts([f"Source: {t.metadata['source']}, Title: {t.metadata['title']} \n\nContent: {t.page_content}" for t in texts], embeddings, index_name=index_name, namespace=namespace)

pc = Pinecone(api_key= pinecone_api_key)

pinecone_index = pc.Index("serenity-sphere")



def perform_rag_fitness(query):
    raw_query_embedding = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )

    query_embedding = raw_query_embedding.data[0].embedding

    # print(" ******    Query Embedding:   ******* ", query_embedding)

    top_matches = pinecone_index.query(vector=query_embedding, top_k=10, include_metadata=True, namespace=namespace) 

    # print(" ***** Top Matches: ***** ", top_matches)

    # Get the list of retrieved texts
    contexts = [item['metadata']['text'] for item in top_matches['matches']]

    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    # Modify the prompt below as need to improve the response quality
    system_prompt = f"""You are an AI fitness coach within SerenitySphere, tasked with helping users achieve their fitness goals through personalized workout plans and motivational support. Your primary goals are to create effective exercise routines, provide guidance, and keep users motivated on their fitness journey. Your responsibilities include:

    Custom Workout Plans: Design personalized workout routines based on user goals, fitness levels, and preferences, updating plans as users progress.

    Exercise Library: Offer a comprehensive library of instructional videos and guides for various exercises, ensuring users perform movements safely and effectively.

    Motivation and Tips: Send daily motivational messages and fitness tips to keep users engaged, inspired, and on track with their goals.

    Goal Setting and Tracking: Assist users in setting realistic fitness goals and provide tools to track progress, celebrating achievements and adjusting plans as needed.

    Health and Wellness Education: Share articles and resources about nutrition, recovery, and overall wellness to complement users' fitness efforts.

    Maintain an enthusiastic and supportive tone, encouraging users to embrace their fitness journey with confidence and commitment.
    """

    res = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ],
        # stream=True
    )

    return res.choices[0].message.content

# ans = perform_rag("What is the video about?")

# print("\n\nANSWER: ", ans)