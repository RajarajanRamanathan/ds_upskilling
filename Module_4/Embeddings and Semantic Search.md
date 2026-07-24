documents = [
    "Artificial intelligence is transforming healthcare.",
    "Deep learning is a subset of machine learning.",
    "Python is widely used for data science.",
    "FAISS enables efficient vector similarity search.",
    "Semantic search finds documents by meaning."
]

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

document_embeddings = model.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print(document_embeddings.shape)