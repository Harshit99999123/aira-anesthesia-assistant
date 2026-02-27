import fitz
from typing import List, Dict


def build_documents_pagewise(pdf_path: str) -> List[Dict]:
    """
    Fallback builder for PDFs without bookmarks.
    Each page becomes one document.
    """

    doc = fitz.open(pdf_path)
    documents = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()

        # Skip very small pages (blank pages)
        if not text.strip():
            continue

        documents.append({
            "hierarchy": [f"Page {page_num + 1}"],
            "level": 1,
            "start_page": page_num,
            "end_page": page_num,
            "text": text
        })

    doc.close()
    return documents