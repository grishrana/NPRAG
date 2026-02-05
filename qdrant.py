import hashlib
from datetime import datetime, timezone
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid


NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")  # any fixed UUID


client = QdrantClient("localhost", port=6333)

if not client.collection_exists("NPRAG"):
    client.create_collection(
        collection_name="NPRAG",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
def embedd_doc():
# Load text
    with open("files/gagan_info.txt", encoding="utf-8") as f:
        text = f.read()

# Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=150,
        separators=["\n\n", "\n", "।", ".", " ", ""],
    )

    chunks = splitter.split_text(text)

    print("Number of chunks:", len(chunks))
    for i,chunk in enumerate(chunks):
        print(f"\n--chunk {i}---")
        print(f"\n--chunklen {len(chunk)}--")
        print(chunk)

    model = SentenceTransformer('BAAI/bge-m3')

    embeddings = model.encode(chunks,batch_size=16, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)

    for embed in embeddings:
        print(embed)

    return embeddings, chunks



def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def chunk_uuid(doc_id: str, chunk_index: int, content_hash: str) -> str:
    name = f"{doc_id}|{chunk_index}|{content_hash}"
    return str(uuid.uuid5(NAMESPACE, name))

# chunk

def build_chunk_metadata(doc_id: str, chunk_index: int, chunk_text: str) -> dict:
    """
    Metadata that is useful for RAG + debugging + cache invalidation.
    """
    content_hash = sha1_hex(chunk_text)
    chunk_id = chunk_uuid(doc_id, chunk_index, content_hash)
    return {
        "chunk_id": chunk_id,                 # stable id
        "doc_id": doc_id,                     # file-based doc id
        "source_path": "/files/gagan_info.txt",
        "chunk_index": chunk_index,           # order
        "text": chunk_text,                   # chunk text (can be large; optional)
        "content_hash": content_hash,         # useful for re-index/invalidation
        "language_hint": "ne",                # hint; can be "mixed" if needed
        "chunking": {
            "method": "RecursiveCharacterTextSplitter",
            "chunk_size": 1024,
            "chunk_overlap": 150,
            "separators": ["\n\n", "\n", "।", ".", " ", ""],
        },
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def create_points():
    embeddings, chunks = embedd_doc()
    points=[]
    for i, (chunk_text, vec) in enumerate(zip(chunks, embeddings)):
        payload = build_chunk_metadata(doc_id="gagan_info.txt", chunk_index=i, chunk_text=chunk_text)
        points.append(
            PointStruct(
                    id=payload["chunk_id"],
                    vector=vec.tolist(),
                    payload=payload,
            )
        )
    
    print(points[:2])
    return points

def upsert():
    points = create_points()
    for start in range(0, len(points), 16):
        batch = points[start:start+16]
        client.upsert(collection_name="NPRAG", points=batch)

def main():
    choice= int(input("1. Create points\n2. Upsert points\n"))
    if choice == 2:
        upsert()

if __name__ == "__main__":
    main()

    
