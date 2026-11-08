import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy

load_dotenv()

client=OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

def get_embedding(text):
    response=client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL"), # type: ignore
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(A,B):
    dotProd=numpy.dot(A,B)
    lenA=numpy.linalg.norm(A)
    lenB=numpy.linalg.norm(B)
    return dotProd/(lenA*lenB)

chunks=["I am a good boy","She is a bad girl","red","blue","black","white","I absolutely love, adore, and cherish this wonderful amazing beautiful day","I absolutely hate, despise, and loathe this terrible horrible disgusting night"]

print(cosine_similarity(get_embedding(chunks[0]),get_embedding(chunks[1])))
print(cosine_similarity(get_embedding(chunks[1]),get_embedding(chunks[2])))
print(cosine_similarity(get_embedding(chunks[-1]),get_embedding(chunks[-2])))
