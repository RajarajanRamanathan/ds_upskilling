import os
import json

import faiss
import numpy as np

from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "gemini-embedding-2"

VECTOR_DIMENSION = 768

TOP_K = 5

VECTOR_STORE_PATH = (
    "vector_store/document.index"
)

METADATA_PATH = (
    "vector_store/metadata.json"
)


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_gemini_client():

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GEMINI_API_KEY environment variable is not set."
        )

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# PAGE-AWARE CHUNKING
# ============================================================

def chunk_pages(
    pages,
    document_name,
    chunk_size=1200,
    overlap=200
):
    """
    Convert page-level text into overlapping chunks.

    Each chunk keeps:
        - document
        - page number
        - chunk id
        - text
    """

    chunks = []

    chunk_id = 0

    for page in pages:

        page_number = page[
            "page_number"
        ]

        text = page[
            "text"
        ].strip()

        if not text:
            continue

        words = text.split()

        start = 0

        while start < len(words):

            end = min(
                start + chunk_size,
                len(words)
            )

            chunk_text = " ".join(
                words[start:end]
            )

            chunks.append({
                "document": document_name,
                "page_number": page_number,
                "chunk_id": chunk_id,
                "text": chunk_text
            })

            chunk_id += 1

            if end >= len(words):
                break

            start = end - overlap

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

def generate_embeddings(texts):

    client = get_gemini_client()

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=VECTOR_DIMENSION
        )
    )

    embeddings = []

    for embedding in result.embeddings:

        embeddings.append(
            embedding.values
        )

    return np.array(
        embeddings,
        dtype="float32"
    )


# ============================================================
# BUILD VECTOR STORE
# ============================================================

def build_vector_store(
    pages,
    document_name
):

    os.makedirs(
        "vector_store",
        exist_ok=True
    )

    chunks = chunk_pages(
        pages,
        document_name
    )

    if not chunks:

        raise ValueError(
            "No usable text was found in the document."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = generate_embeddings(
        texts
    )

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(
        embeddings
    )

    index = faiss.IndexFlatIP(
        VECTOR_DIMENSION
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        VECTOR_STORE_PATH
    )

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    return len(chunks)


# ============================================================
# SEARCH
# ============================================================

def search_documents(
    query,
    top_k=TOP_K
):

    if not os.path.exists(
        VECTOR_STORE_PATH
    ):

        raise ValueError(
            "Knowledge base does not exist. "
            "Build the knowledge base first."
        )

    if not os.path.exists(
        METADATA_PATH
    ):

        raise ValueError(
            "Metadata file does not exist."
        )

    client = get_gemini_client()

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            output_dimensionality=VECTOR_DIMENSION
        )
    )

    query_embedding = np.array(
        [result.embeddings[0].values],
        dtype="float32"
    )

    faiss.normalize_L2(
        query_embedding
    )

    index = faiss.read_index(
        VECTOR_STORE_PATH
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(
            file
        )

    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal)
    )

    results = []

    for score, index_number in zip(
        scores[0],
        indices[0]
    ):

        if index_number < 0:
            continue

        chunk = metadata[
            index_number
        ].copy()

        chunk["similarity"] = float(
            score
        )

        results.append(
            chunk
        )

    return results


# ============================================================
# QUESTION ANSWERING
# ============================================================

def answer_question(question):

    retrieved_chunks = search_documents(
        question
    )

    if not retrieved_chunks:

        return {
            "answer": (
                "I could not find relevant information "
                "in the document."
            ),
            "sources": []
        }

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"""
Document: {chunk['document']}
Page: {chunk['page_number']}
Chunk ID: {chunk['chunk_id']}

Content:
{chunk['text']}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are an AI document assistant.

Answer the user's question ONLY using
the supplied document context.

If the answer cannot be found in the
context, clearly say that the information
was not found.

Do not invent information.

USER QUESTION:
{question}

DOCUMENT CONTEXT:
{context}

Return JSON:

{{
    "answer": "...",
    "sources": [
        {{
            "document": "...",
            "page_number": 0,
            "chunk_id": 0,
            "evidence": "..."
        }}
    ]
}}
"""

    client = get_gemini_client()

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    response_text = (
        response.text
        .strip()
    )

    # Remove markdown fences if Gemini adds them
    if response_text.startswith(
        "```json"
    ):

        response_text = (
            response_text[7:]
            .strip()
        )

    if response_text.endswith(
        "```"
    ):

        response_text = (
            response_text[:-3]
            .strip()
        )

    try:

        return json.loads(
            response_text
        )

    except json.JSONDecodeError:

        return {
            "answer": response_text,
            "sources": [
                {
                    "document": chunk[
                        "document"
                    ],
                    "page_number": chunk[
                        "page_number"
                    ],
                    "chunk_id": chunk[
                        "chunk_id"
                    ],
                    "evidence": chunk[
                        "text"
                    ][:500]
                }
                for chunk in retrieved_chunks
            ]
        }