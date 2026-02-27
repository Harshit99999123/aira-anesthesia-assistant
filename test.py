from ingestion.bookmark_parser import parse_pdf_bookmarks
from ingestion.document_builder import build_documents_with_text
from ingestion.chunker import chunk_documents
from vectorstore.build_vectorstore import VectorStoreBuilder

PDF_PATH = "The-ICU-Book-5E-2025-Paul-L-Marino-ALGrawany.pdf"


def main():
    print("Step 1: Parsing bookmarks...")
    tree = parse_pdf_bookmarks(PDF_PATH)

    print("Step 2: Filtering only Volume nodes...")
    volume_nodes = [node for node in tree if node.title.startswith("Volume")]

    print("Step 3: Building structured documents with text...")
    docs = build_documents_with_text(PDF_PATH, volume_nodes)

    print("Total structured documents:", len(docs))

    print("Step 4: Chunking documents...")
    chunked_docs = chunk_documents(docs)

    print("Total chunks:", len(chunked_docs))

    if chunked_docs:
        print("\n--- SAMPLE CHUNK METADATA ---")
        sample = chunked_docs[0]
        print("Book:", sample["book"])
        print("Volume:", sample["volume"])
        print("Section:", sample["section"])
        print("Chapter:", sample["chapter"])
        print("Heading:", sample["heading"])
        print("Pages:", sample["start_page"], "-", sample["end_page"])
        print("Chunk Index:", sample["chunk_index"])

        print("\n--- SAMPLE CHUNK TEXT (first 500 chars) ---")
        print(sample["text"][:500])

    builder = VectorStoreBuilder(
        persist_directory="vectorstore",
        rebuild=True  # Set False after first build
    )

    builder.embed_and_store(chunked_docs)
    from retrieval.retriever import Retriever

    retriever = Retriever(
        similarity_threshold=0.4,
        top_k=5
    )

    query = "What are the effects of inhaled anesthetics on cerebral blood flow?"

    response = retriever.retrieve(query)

    print(response["status"])

    if response["status"] == "success":
        print("Max similarity:", response["max_similarity"])
        print("\nTop result metadata:")
        print(response["results"][0]["metadata"])
        print("\nSnippet:")
        print(response["results"][0]["text"][:500])
    else:
        print("Similarity score:", response["similarity"])


if __name__ == "__main__":
    main()