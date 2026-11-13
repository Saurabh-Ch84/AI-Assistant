import os, numpy, json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

EMBED_MODEL = os.getenv("EMBEDDING_MODEL")
KB_FOLDER = os.getenv("KB_FOLDER")
VECTOR_STORE_FILE = os.getenv("VECTOR_STORE_FILE")

def get_embedding(text):
    response = client.embeddings.create(
        model=EMBED_MODEL,  # type: ignore
        input=text
    )
    return response.data[0].embedding 

def cosine_similarity(A, B):
    dotProd = numpy.dot(A, B)
    lenA = numpy.linalg.norm(A)
    lenB = numpy.linalg.norm(B)
    if lenA == 0 or lenB == 0:
        return 0.0
    return float(dotProd / (lenA * lenB))

def load_documents(folder=KB_FOLDER):
    documents = []
    if folder is None:
        return documents
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            documents.append({
                "source": filename,
                "text": f.read()
            })
    return documents

def add_embeddings(chunks):
    for chunk in chunks:
        chunk["vector"] = get_embedding(chunk["text"])
    return chunks

def chunk_document(document):
    chunks = []
    chunk_id = 0
    for chunk in document["text"].split("\n\n"):
        chunk = chunk.strip()
        if chunk:
            chunks.append({
                "source": document["source"],
                "chunk_id": chunk_id,
                "text": chunk
            })
            chunk_id += 1
    return chunks

def create_chunks():
    documents = load_documents()
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document))
    return chunks

def load_vector_store(file=VECTOR_STORE_FILE):
    if file is None:
        raise ValueError("VECTOR_STORE_FILE must be configured")
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_vector_store(chunks, file=VECTOR_STORE_FILE):
    if file is None:
        raise ValueError("VECTOR_STORE_FILE must be configured")
    with open(file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4)

def load_chunks():
    if VECTOR_STORE_FILE is None:
        raise ValueError("VECTOR_STORE_FILE must be configured")
    if os.path.exists(VECTOR_STORE_FILE):
        chunks = load_vector_store()
        print(f"Loaded {len(chunks)} chunks from the file {VECTOR_STORE_FILE}")
    else:
        chunks = create_chunks()
        chunks = add_embeddings(chunks)
        save_vector_store(chunks)
        print(f"Created and embedded {len(chunks)} chunks and saved to {VECTOR_STORE_FILE}")
    return chunks

chunks = load_chunks()

def search_top_k(query, top_k=3):
    query_vector = get_embedding(query)
    results = []
    for chunk in chunks:
        score = cosine_similarity(query_vector, chunk["vector"])
        results.append({
            "score": float(score),
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "size": len(chunk["text"])
        })

    results.sort(key=lambda rec: rec["score"], reverse=True)
    return results[:top_k]