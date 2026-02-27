import fitz
from typing import List, Dict
from model.bookmark_node import BookmarkNode


def extract_text_from_range(doc, start_page: int, end_page: int) -> str:
    text_parts = []

    for page_num in range(start_page, end_page + 1):
        page = doc.load_page(page_num)
        text_parts.append(page.get_text())

    return "\n".join(text_parts)


def build_documents_with_text(
    pdf_path: str,
    nodes: List[BookmarkNode],
    book_name: str = "Miller Anesthesia"
) -> List[Dict]:

    documents = []
    doc = fitz.open(pdf_path)

    def traverse(node_list, current_volume=None, current_section=None, current_chapter=None):
        for node in node_list:

            if node.level == 1 and node.title.startswith("Volume"):
                current_volume = node.title

            elif node.level == 2:
                current_section = node.title

            elif node.level == 3:
                current_chapter = node.title

            if not node.children:
                text = extract_text_from_range(doc, node.start_page, node.end_page)

                documents.append({
                    "book": book_name,
                    "volume": current_volume,
                    "section": current_section,
                    "chapter": current_chapter,
                    "heading": node.title,
                    "start_page": node.start_page,
                    "end_page": node.end_page,
                    "text": text
                })
            else:
                traverse(
                    node.children,
                    current_volume=current_volume,
                    current_section=current_section,
                    current_chapter=current_chapter
                )

    traverse(nodes)
    doc.close()

    return documents