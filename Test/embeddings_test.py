import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client=OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

chunks=["I am a good boy","She loves cock","She sucks dick","penis","cock","dick","banana","pussy","red"]

response=client.embeddings.create(
    model=os.getenv("EMBEDDING_MODEL"), # type: ignore
    input=chunks
)

for i, embed in enumerate(response.data):
    print(f"Embedding {i} length: {len(embed.embedding)}")
    print(embed.embedding[:5])