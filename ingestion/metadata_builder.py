# from typing import List, Dict
# from model.bookmark_node import BookmarkNode
#
#
# def build_enriched_documents(
#     nodes: List[BookmarkNode],
#     book_name: str = "Miller Anesthesia"
# ) -> List[Dict]:
#     """
#     Traverse bookmark tree and build enriched metadata
#     only for leaf nodes.
#     """
#
#     documents = []
#
#     def traverse(node_list, current_volume=None, current_section=None, current_chapter=None):
#         for node in node_list:
#
#             # Determine hierarchy by level
#             if node.level == 1 and node.title.startswith("Volume"):
#                 current_volume = node.title
#
#             elif node.level == 2:
#                 current_section = node.title
#
#             elif node.level == 3:
#                 current_chapter = node.title
#
#             # Leaf node = actual content unit
#             if not node.children:
#                 documents.append({
#                     "book": book_name,
#                     "volume": current_volume,
#                     "section": current_section,
#                     "chapter": current_chapter,
#                     "heading": node.title,
#                     "start_page": node.start_page,
#                     "end_page": node.end_page,
#                 })
#             else:
#                 traverse(
#                     node.children,
#                     current_volume=current_volume,
#                     current_section=current_section,
#                     current_chapter=current_chapter
#                 )
#
#     traverse(nodes)
#
#     return documents