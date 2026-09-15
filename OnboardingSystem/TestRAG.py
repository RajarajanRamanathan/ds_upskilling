from services.rag_service_FAISS import (
    build_vector_store,
    search_documents
)

document = """
The project will be completed within 90 days.

The vendor will provide application development
and testing services.

The customer will provide business requirements.

Support will be provided for 30 days after deployment.

Either party may terminate the agreement with
30 days written notice.
"""


chunks = build_vector_store(
    document,
    "sample_sow.txt"
)

print(
    f"Created {chunks} chunks"
)


results = search_documents(
    "What is the termination notice period?"
)

for result in results:

    print(
        "\nScore:",
        result["score"]
    )

    print(
        "Text:",
        result["text"]
    )