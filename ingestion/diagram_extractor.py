import os
from typing import Dict, List, Optional

import fitz


def _extract_page_diagrams(
    page: fitz.Page,
    output_dir: str,
    min_dimension: int = 100,
    max_images_per_page: int = 12,
) -> List[str]:
    """
    Extract raster images from a single PDF page and save them to disk.
    """
    os.makedirs(output_dir, exist_ok=True)

    page_num = page.number + 1
    saved_paths: List[str] = []

    images = page.get_images(full=True)
    for index, image in enumerate(images):
        if len(saved_paths) >= max_images_per_page:
            break

        xref = image[0]
        width = image[2] if len(image) > 2 else 0
        height = image[3] if len(image) > 3 else 0

        # Skip tiny icons/noise.
        if width < min_dimension or height < min_dimension:
            continue

        try:
            image_data = page.parent.extract_image(xref)
        except Exception:
            continue

        img_bytes = image_data.get("image")
        ext = image_data.get("ext", "png")
        if not img_bytes:
            continue

        file_name = f"page_{page_num:04d}_img_{index + 1:02d}.{ext}"
        path = os.path.abspath(os.path.join(output_dir, file_name))

        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(img_bytes)

        saved_paths.append(path)

    if saved_paths:
        return saved_paths

    # Fallback for vector-drawn diagrams (common in textbooks):
    # if page text suggests a figure/diagram, render page as an image.
    page_text = (page.get_text() or "").lower()
    has_figure_hint = any(
        token in page_text
        for token in ("figure", "fig.", "fig ", "diagram", "algorithm")
    )

    if has_figure_hint:
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        preview_name = f"page_{page_num:04d}_render.png"
        preview_path = os.path.abspath(os.path.join(output_dir, preview_name))
        if not os.path.exists(preview_path):
            pix.save(preview_path)
        saved_paths.append(preview_path)

    return saved_paths


def extract_diagrams_for_range(
    doc: fitz.Document,
    start_page: int,
    end_page: int,
    output_dir: Optional[str] = None,
    page_diagram_cache: Optional[Dict[int, List[str]]] = None,
    max_diagrams_per_range: int = 20,
) -> List[str]:
    """
    Extract diagrams for a PDF page range (inclusive), with page-level cache.
    """
    if not output_dir:
        return []

    cache = page_diagram_cache if page_diagram_cache is not None else {}
    diagrams: List[str] = []

    for page_index in range(start_page, end_page + 1):
        if page_index not in cache:
            page = doc.load_page(page_index)
            cache[page_index] = _extract_page_diagrams(page, output_dir)

        for path in cache[page_index]:
            if path not in diagrams:
                diagrams.append(path)
                if len(diagrams) >= max_diagrams_per_range:
                    return diagrams

    return diagrams
