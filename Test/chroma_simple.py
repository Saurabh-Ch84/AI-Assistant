import chromadb, os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client=OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

EMBED_MODEL=os.getenv("EMBEDDING_MODEL")

def get_embedding(text):
    response=client.embeddings.create(
        model=EMBED_MODEL, # type: ignore
        input=text
    )
    return response.data[0].embedding

chunks={
    "c0": "Leave Policy - Maternity Leave\n Maternity leave is 26 weeks of paid leave for the first two children.",
    "c1":"Medical Policy - Maternity Cover\n Hospitalization for childbirth is covered up to 75,000 rupees.",
    "c2":"Travel Expense - Reimbursement\n Submit expense claims within 30 days of the trip with all receipts attached. Approved claims are reimbursed within 7 working days.",
    "c3":"Tech Docs - Deployment Windows\n Production deployments are allowed on Tuesdays and Thursdays between 11 AM and 4 PM. There are no deployments on Fridays."
}

ids=list(chunks.keys())
documents=list(chunks.values())

embeddings=[]
for d in documents:
    embeddings.append(get_embedding(d))


chroma=chromadb.Client()

zorvex_collection=chroma.create_collection(
    name="zorvex_technology",
    configuration={
        "hnsw":{
            "space":"cosine"
        }
    }
)

zorvex_collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

query="When is the production deployment allowed?"
query_embedding=get_embedding(query)

result=zorvex_collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

print(result["documents"][0]) # type: ignore