import os, numpy
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client=OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

EMBED_MODEL=os.getenv("EMBEDDING_MODEL")
KB_FOLDER=os.getenv("KB_FOLDER")

def get_embedding(text):
    response=client.embeddings.create(
        model=EMBED_MODEL, # type: ignore
        input=text
    )
    return response.data[0].embedding 

def cosine_similarity(A,B):
    dotProd=numpy.dot(A,B)
    lenA=numpy.linalg.norm(A)
    lenB=numpy.linalg.norm(B)
    return dotProd/(lenA*lenB)

def load_documents(folder=KB_FOLDER):
    documents=[]
    if folder is None:
        return documents
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue
        path=os.path.join(folder,filename)
        with open(path,"r",encoding="utf-8") as f:
            documents.append(
                {
                    "source":filename,
                    "text": f.read()
                }
            )
    return documents

def add_embeddings(documents):
    for document in documents:
        document["vector"]=get_embedding(document["text"])
    return documents

def search_top_k(query, documents, top_k=3):
    query_vector=get_embedding(query)
    results=[]
    for document in documents:
        score=cosine_similarity(query_vector,document["vector"])
        results.append({
            "score":float(score),
            "source":document["source"],
            "text":document["text"],
            "size":len(document["text"])
        })

    results.sort(key=lambda rec: rec["score"], reverse=True)
    return results[:top_k]

def test(user_query, documents, top_k=3):
    print(f"\n User Query: {user_query}\n")
    results=search_top_k(user_query,documents,top_k)
    for r in results:
        print(f"Score: {r["score"]:.2f}, Source: {r["source"]}, Size: {r["size"]} chars")

documents=load_documents()
documents=add_embeddings(documents)
print(f"We have embedded {len(documents)} docs")
user_query="How many days I can get for maternity leave?"
test(user_query,documents,3)