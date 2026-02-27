import fitz
from typing import List
from model.bookmark_node import BookmarkNode


def extract_text_from_range(doc, start_page: int, end_page: int) -> str:
    """
    Extract text from PDF between start_page and end_page (inclusive).
    """
    text_parts = []

    for page_num in range(start_page, end_page + 1):
        page = doc.load_page(page_num)
        text_parts.append(page.get_text())

    return "\n".join(text_parts)


def extract_leaf_nodes_text(pdf_path: str, nodes: List[BookmarkNode]) -> List[dict]:
    """
    Traverse tree and extract text only from leaf nodes.
    Returns list of dicts containing:
        - title
        - level
        - start_page
        - end_page
        - text
    """
    doc = fitz.open(pdf_path)
    extracted_data = []

    def traverse(node_list: List[BookmarkNode]):
        for node in node_list:
            if not node.children:  # Leaf node
                text = extract_text_from_range(doc, node.start_page, node.end_page)

                extracted_data.append({
                    "title": node.title,
                    "level": node.level,
                    "start_page": node.start_page,
                    "end_page": node.end_page,
                    "text": text
                })
            else:
                traverse(node.children)

    traverse(nodes)
    doc.close()

    return extracted_data