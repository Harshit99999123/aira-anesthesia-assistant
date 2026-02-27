import fitz  # PyMuPDF
from typing import List
from model.bookmark_node import BookmarkNode


def build_bookmark_tree(toc: List[List]) -> List[BookmarkNode]:
    """
    Convert flat TOC list into hierarchical BookmarkNode tree.
    """
    stack: List[BookmarkNode] = []
    root_nodes: List[BookmarkNode] = []

    for level, title, page in toc:
        node = BookmarkNode(
            level=level,
            title=title.strip(),
            start_page=page - 1  # convert to 0-based index
        )

        # Pop stack until we find correct parent
        while stack and stack[-1].level >= level:
            stack.pop()

        if stack:
            stack[-1].add_child(node)
        else:
            root_nodes.append(node)

        stack.append(node)

    return root_nodes


def assign_end_pages(nodes: List[BookmarkNode], total_pages: int):
    """
    Assign end_page using hierarchy-aware logic.
    """

    def assign_recursive(node_list: List[BookmarkNode], parent_end: int):
        for i, node in enumerate(node_list):

            # Determine boundary using next sibling
            if i < len(node_list) - 1:
                next_node = node_list[i + 1]
                node.end_page = next_node.start_page - 1
            else:
                node.end_page = parent_end

            # Recursively assign for children
            if node.children:
                assign_recursive(node.children, node.end_page)

    assign_recursive(nodes, total_pages - 1)


def remove_invalid_nodes(nodes: List[BookmarkNode]) -> List[BookmarkNode]:
    """
    Remove nodes where start_page > end_page.
    These are usually navigation-only bookmarks.
    """

    cleaned_nodes: List[BookmarkNode] = []

    for node in nodes:
        # Clean children first
        node.children = remove_invalid_nodes(node.children)

        # Keep node only if valid page range
        if node.start_page is not None and node.end_page is not None:
            if node.start_page <= node.end_page:
                cleaned_nodes.append(node)
        else:
            cleaned_nodes.append(node)

    return cleaned_nodes


def parse_pdf_bookmarks(pdf_path: str) -> list[BookmarkNode] | None:
    """
    Main function:
    - Opens PDF
    - Extracts TOC
    - Builds hierarchical tree
    - Assigns page ranges
    - Removes invalid bookmark ranges
    """

    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    total_pages = len(doc)

    if not toc:
        return None

    tree = build_bookmark_tree(toc)
    assign_end_pages(tree, total_pages)
    tree = remove_invalid_nodes(tree)

    doc.close()

    return tree