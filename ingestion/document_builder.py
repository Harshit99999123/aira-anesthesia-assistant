import fitz
from typing import List, Dict
from model.bookmark_node import BookmarkNode


def extract_text_from_range(doc, start_page: int, end_page: int) -> str:
    """
    Extract text from a page range (inclusive).
    """
    text_parts = []

    for page_num in range(start_page, end_page + 1):
        page = doc.load_page(page_num)
        text_parts.append(page.get_text())

    return "\n".join(text_parts)


def build_documents_with_text(
    pdf_path: str,
    nodes: List[BookmarkNode],
) -> List[Dict]:
    """
    Build structured documents from bookmark tree.

    Each leaf node becomes one document.

    Returns:
        List[Dict] with:
            - hierarchy (full path from root to leaf)
            - level (leaf level)
            - start_page
            - end_page
            - text
    """

    documents: List[Dict] = []
    doc = fitz.open(pdf_path)

    def traverse(node_list: List[BookmarkNode], hierarchy_path: List[str]):
        for node in node_list:

            current_path = hierarchy_path + [node.title]

            # Leaf node → create document
            if not node.children:
                text = extract_text_from_range(
                    doc,
                    node.start_page,
                    node.end_page
                )

                documents.append({
                    "hierarchy": current_path,
                    "level": node.level,
                    "start_page": node.start_page,
                    "end_page": node.end_page,
                    "text": text
                })

            # Non-leaf → continue traversal
            else:
                traverse(node.children, current_path)

    traverse(nodes, [])
    doc.close()

    return documents