import os
import shutil
from ingestion.bookmark_parser import parse_pdf_bookmarks
from ingestion.document_builder import build_documents_with_text
from ingestion.chunker import chunk_documents
from ingestion.page_builder import build_documents_pagewise
from vectorstore.build_vectorstore import VectorStoreBuilder


PDF_PATH = "/Users/harshit/PycharmProjects/anaestheia_assisstant/data_bank/The-ICU-Book-5E-2025-Paul-L-Marino-ALGrawany.pdf"


def generate_book_id(pdf_path: str) -> str:
    filename = os.path.basename(pdf_path)
    return filename.replace(".pdf", "").lower().replace(" ", "_")


def start_ingestion():
    book_id = generate_book_id(PDF_PATH)
    book_name = os.path.basename(PDF_PATH).replace(".pdf", "")
    diagrams_dir = os.path.abspath(os.path.join("data_bank", "diagrams", book_id))

    print("Book ID:", book_id)
    print("Book Name:", book_name)
    print("Diagrams Dir:", diagrams_dir)

    # Remove previous diagram extracts for this book.
    if os.path.exists(diagrams_dir):
        shutil.rmtree(diagrams_dir)

    print("\nStep 1: Parsing bookmarks...")
    tree = parse_pdf_bookmarks(PDF_PATH)

    print("Step 2: Building structured documents with text...")
    if tree:
        docs = build_documents_with_text(
            PDF_PATH,
            tree,
            diagrams_output_dir=diagrams_dir
        )
    else:
        print("No bookmarks found. Falling back to page-wise ingestion.")
        docs = build_documents_pagewise(
            PDF_PATH,
            diagrams_output_dir=diagrams_dir
        )

    print("Total structured documents:", len(docs))

    print("\nStep 3: Chunking documents...")
    chunked_docs = chunk_documents(docs)

    print("Total chunks:", len(chunked_docs))

    if chunked_docs:
        sample = chunked_docs[0]
        print("\n--- SAMPLE CHUNK METADATA ---")
        print("Hierarchy:", " → ".join(sample["hierarchy"]))
        print("Pages:", sample["start_page"], "-", sample["end_page"])
        print("Chunk Index:", sample["chunk_index"])
        print("Diagram Count:", len(sample.get("diagram_paths", [])))

        print("\n--- SAMPLE CHUNK TEXT (first 500 chars) ---")
        print(sample["text"][:500])

    print("\nStep 4: Embedding and storing...")

    builder = VectorStoreBuilder(
        persist_directory="vectorstore"
    )

    builder.embed_and_store(
        chunked_docs=chunked_docs,
        book_id=book_id,
        book_name=book_name
    )

    print("\nIngestion complete.")

    # --------------------------------------------------
    # Optional: Test retrieval
    # --------------------------------------------------

    from retrieval.retriever import Retriever

    retriever = Retriever(
        similarity_threshold=0.4,
        top_k=5
    )

    query = "What are the effects of inhaled anesthetics on cerebral blood flow?"

    print("\nTesting retrieval...\n")

    response = retriever.retrieve(query, book_id=book_id)

    print("Status:", response["status"])

    if response["status"] == "success":
        print("Max similarity:", response["max_similarity"])
        print("\nTop result metadata:")
        print(response["results"][0]["metadata"])
        print("\nSnippet:")
        print(response["results"][0]["text"][:500])
    else:
        print("Similarity score:", response["similarity"])


if __name__ == "__main__":
    start_ingestion()
