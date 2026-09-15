import os
import chromadb

from google import genai
from google.genai import types


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBEDDING_MODEL = "gemini-embedding-2"
LLM_MODEL = "gemini-3.5-flash"

VECTOR_DIMENSION = 768
TOP_K = 5

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "Onboarding_Documents"

# ---------------------------------------------------------
# Gemini client
# ---------------------------------------------------------

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

# ---------------------------------------------------------
# Chroma client
# ---------------------------------------------------------

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    configuration={
        "hnsw": {
            "space": "cosine"
        }
    }
)


# ---------------------------------------------------------
# Chunking
# ---------------------------------------------------------

def chunk_pages(
    pages,
    document_name,
    chunk_size=1200,
    overlap=200
):
    """
    Split page-aware document content into chunks while
    preserving document and page information.
    """

    chunks = []

    for page in pages:

        page_number = page["page_number"]
        text = page["text"].strip()

        if not text:
            continue

        start = 0
        chunk_id = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            if chunk_text.strip():

                chunks.append({
                    "document": document_name,
                    "page_number": page_number,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "source": page.get("source", "text")
                })

                chunk_id += 1

            start = end - overlap

            if start < 0:
                start = 0

            if end >= len(text):
                break

    return chunks


# ---------------------------------------------------------
# Gemini Embeddings
# ---------------------------------------------------------

def generate_embeddings(texts):
    """
    Generate Gemini embeddings for a list of texts.

    Gemini embedding-2 supports reduced output dimensions.
    We use 768 dimensions to keep storage and retrieval
    efficient.
    """

    embeddings = []

    for text in texts:

        result = gemini_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=VECTOR_DIMENSION
            )
        )

        embedding = result.embeddings[0].values

        embeddings.append(embedding)

    return embeddings


# ---------------------------------------------------------
# Build Chroma knowledge base
# ---------------------------------------------------------

def build_vector_store(
    pages,
    document_name
):
    """
    Build/update the Chroma knowledge base.

    Returns:
        Number of indexed chunks.
    """

    chunks = chunk_pages(
        pages,
        document_name
    )

    if not chunks:
        raise ValueError(
            "No usable text was found in the document."
        )

    # Remove existing chunks belonging to this document.
    try:
        collection.delete(
            where={
                "document": document_name
            }
        )
    except Exception:
        pass

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = generate_embeddings(texts)

    ids = []

    metadatas = []

    for index, chunk in enumerate(chunks):

        chunk_id = (
            f"{document_name}"
            f"_page_{chunk['page_number']}"
            f"_chunk_{index}"
        )

        ids.append(chunk_id)

        metadatas.append({
            "document": chunk["document"],
            "page_number": chunk["page_number"],
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"]
        })

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)


# ---------------------------------------------------------
# Search Chroma
# ---------------------------------------------------------

def search_documents(
    query,
    top_k=TOP_K
):
    """
    Search Chroma for semantically similar document chunks.
    """

    query_embedding = generate_embeddings(
        [query]
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    search_results = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        search_results.append({
            "document": metadata.get(
                "document",
                ""
            ),
            "page_number": metadata.get(
                "page_number",
                0
            ),
            "chunk_id": metadata.get(
                "chunk_id",
                0
            ),
            "text": document,
            "similarity": 1 - distance
        })

    return search_results


# ---------------------------------------------------------
# RAG Question Answering
# ---------------------------------------------------------

def answer_question(question):

    retrieved_chunks = search_documents(
        question,
        TOP_K
    )

    if not retrieved_chunks:
        return {
            "answer": (
                "I could not find relevant information "
                "in the indexed documents."
            ),
            "sources": []
        }

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"""
Document: {chunk['document']}
Page: {chunk['page_number']}
Chunk: {chunk['chunk_id']}

Content:
{chunk['text']}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an AI assistant helping review
SOWs, contracts and SLAs.

Answer the user's question using ONLY
the supplied document context.

If the answer cannot be determined from
the context, clearly say so.

Do not invent contract terms.

USER QUESTION:
{question}

DOCUMENT CONTEXT:
{context}

Return valid JSON:

{{
    "answer": "Answer based on the document",
    "sources": [
        {{
            "document": "Document name",
            "page_number": 1,
            "chunk_id": 1,
            "evidence": "Relevant evidence"
        }}
    ]
}}
"""

    response = gemini_client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    response_text = response.text.strip()

    # Remove markdown JSON fences if Gemini adds them.
    if response_text.startswith("```json"):
        response_text = response_text[
            len("```json"):
        ].strip()

    if response_text.endswith("```"):
        response_text = response_text[
            :-3
        ].strip()

    try:
        import json

        result = json.loads(response_text)

        return result

    except Exception:

        return {
            "answer": response_text,
            "sources": [
                {
                    "document": chunk["document"],
                    "page_number": chunk["page_number"],
                    "chunk_id": chunk["chunk_id"],
                    "evidence": chunk["text"]
                }
                for chunk in retrieved_chunks
            ]
        }