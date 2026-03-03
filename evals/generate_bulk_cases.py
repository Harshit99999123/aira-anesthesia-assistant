import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings


STOPWORDS = {
    "about", "above", "after", "again", "against", "all", "also", "among", "an", "and", "any",
    "are", "around", "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "could", "did", "do", "does", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
    "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "why", "will", "with", "you", "your",
}

GENERIC_HEADINGS = {
    "cover", "cover image", "title page", "copyright", "dedication", "contents", "table of contents",
    "contributors", "index", "preface", "acknowledgments", "instructions for online access",
}

DOMAIN_ANCHORS = {
    "anesthesia", "anaesthesia", "analgesia", "airway", "intubation", "ventilation", "oxygenation",
    "sedation", "hemodynamic", "haemodynamic", "hypotension", "hypertension", "vasopressor", "shock",
    "sepsis", "icu", "critical", "perioperative", "postoperative", "anesthetic", "respiratory", "cardiac",
    "renal", "hepatic", "neurologic", "coagulation", "fluid", "electrolyte", "pain", "regional", "spinal",
}

NOISE_TERMS = {
    "et", "al", "doi", "isbn", "copyright", "references", "appendix", "chapter", "table", "figure",
    "fig", "page", "pages", "volume", "edition", "online", "access", "preface", "index", "contributors",
}

DISCOURSE_TERMS = {
    "however", "important", "experts", "contrast", "well", "section", "improving", "direct", "mean",
    "means", "provide", "provides", "using", "used", "usually", "daily", "still", "result", "effects",
    "effect", "introduction", "advantages", "considerations",
}

BAD_TOPIC_PATTERNS = [
    r"\bassistant professor\b",
    r"\bdepartment of\b",
    r"\bmedical school\b",
    r"\bcontributor\b",
    r"\bcontributors\b",
    r"\bvideo contents\b",
    r"\bclinical professor\b",
    r"\buniversity of\b",
    r"\bmd\b",
    r"\bfasa\b",
]

QUESTION_TEMPLATES = [
    "Explain the clinical significance of {topic}.",
    "What are the key mechanisms and management principles for {topic}?",
    "Summarize diagnosis, monitoring, and treatment considerations for {topic}.",
    "How does {topic} affect perioperative or critical-care decision making?",
    "Give a structured review of {topic} with practical bedside implications.",
]


@dataclass
class Candidate:
    book_id: str
    book_name: str
    topic: str
    query: str
    expected_terms: List[str]
    has_diagram: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate large eval case set from indexed books")
    parser.add_argument("--output", default="evals/cases_bulk.json", help="Output JSON path")
    parser.add_argument("--persist-directory", default="vectorstore", help="Chroma persist directory")
    parser.add_argument("--collection-name", default="medical_knowledge", help="Chroma collection name")
    parser.add_argument("--per-book", type=int, default=160, help="Number of cases per book")
    parser.add_argument(
        "--diagram-cases-miller",
        type=int,
        default=80,
        help="Minimum number of Miller cases that require diagrams",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _safe_topic(text: str) -> str:
    text = _clean_text(text)
    text = re.sub(r"[^A-Za-z0-9\-\s/(),]", "", text)
    text = re.sub(r"\s+", " ", text).strip(" .,-")
    return text[:120]


def _normalize_topic(topic: str) -> str:
    t = _clean_text(topic)
    t = re.sub(r"^(however|in contrast|therefore|thus|overall|notably)[,\\s]+", "", t, flags=re.I)
    t = re.sub(r"^(it is important to|it is important that)\\s+", "", t, flags=re.I)
    t = re.sub(r"\\s+", " ", t).strip(" .,-")
    return t[:120]


def _parse_hierarchy(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
        return [s]
    return [str(raw).strip()]


def _is_generic_heading(heading: str) -> bool:
    h = heading.lower().strip()
    if h in GENERIC_HEADINGS:
        return True
    if re.fullmatch(r"page\s*\d+", h):
        return True
    return len(h) <= 2


def _is_noisy_chunk(text: str) -> bool:
    lower = text.lower()
    year_count = sum(lower.count(str(y)) for y in range(1950, 2035))
    if year_count >= 4:
        return True
    if lower.count(";") > 8:
        return True
    if any(mark in lower for mark in ["et al", "doi", "http://", "https://"]):
        return True
    return False


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [_clean_text(p) for p in parts if _clean_text(p)]


def _sentence_score(sentence: str) -> int:
    tokens = _tokenize_words(sentence)
    if len(tokens) < 6 or len(tokens) > 24:
        return -100

    noise_hits = sum(1 for t in tokens if t in NOISE_TERMS)
    anchor_hits = sum(1 for t in tokens if t in DOMAIN_ANCHORS)
    long_good = sum(1 for t in tokens if len(t) >= 7 and t not in STOPWORDS and t not in NOISE_TERMS)

    if noise_hits >= 3:
        return -50

    if re.search(r"\b(et al|doi|isbn|www|http)\b", sentence.lower()):
        return -80

    digit_chars = sum(1 for ch in sentence if ch.isdigit())
    if digit_chars > max(3, len(sentence) * 0.08):
        return -40

    return anchor_hits * 5 + long_good - noise_hits * 3


def _extract_topic(hierarchy: List[str], text: str) -> Optional[str]:
    for item in reversed(hierarchy):
        candidate = _safe_topic(item)
        if candidate and not _is_generic_heading(candidate):
            return candidate

    text = _clean_text(text)[:1400]
    if not text:
        return None

    sentences = _split_sentences(text)
    scored: List[Tuple[int, str]] = []
    for sent in sentences[:8]:
        score = _sentence_score(sent)
        if score > 0:
            scored.append((score, sent))

    if not scored:
        return None

    best = sorted(scored, key=lambda x: x[0], reverse=True)[0][1]
    topic = _safe_topic(best)
    topic = _normalize_topic(topic)
    if len(_tokenize_words(topic)) < 4:
        return None
    return topic


def _extract_terms(topic: str, token_df: Counter) -> List[str]:
    tokens = _tokenize_words(topic)
    picked: List[str] = []

    for tok in tokens:
        if tok in DOMAIN_ANCHORS and tok not in picked:
            picked.append(tok)
        if len(picked) >= 2:
            return picked[:2]

    for tok in tokens:
        if tok in STOPWORDS or tok in NOISE_TERMS:
            continue
        if tok in DISCOURSE_TERMS:
            continue
        if not re.search(r"[aeiou]", tok):
            continue
        if len(tok) < 4 or len(tok) > 18:
            continue

        df = token_df.get(tok, 0)
        # Drop very rare OCR/name artifacts and overly generic high-frequency terms.
        if df < 3 or df > 2500:
            continue

        if tok not in picked:
            picked.append(tok)
        if len(picked) >= 2:
            break

    return picked


def _is_bad_topic(topic: str) -> bool:
    lower = topic.lower()
    if any(re.search(pat, lower) for pat in BAD_TOPIC_PATTERNS):
        return True
    tokens = _tokenize_words(lower)
    if len(tokens) < 3:
        return True
    if lower.startswith("it ") or lower.startswith("this "):
        return True
    return False


def _has_diagram(raw) -> bool:
    if isinstance(raw, list):
        return any(isinstance(x, str) and x.strip() for x in raw)
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return False
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return any(isinstance(x, str) and x.strip() for x in arr)
        except Exception:
            pass
        return True
    return False


def _source_hints(book_id: str, book_name: str) -> List[str]:
    hints = {book_id.lower(), book_name.lower()}
    if "miller" in book_id.lower() or "miller" in book_name.lower():
        hints.add("miller")
    if "barash" in book_id.lower() or "barash" in book_name.lower():
        hints.add("barash")
    if "icu" in book_id.lower() or "marino" in book_name.lower():
        hints.add("icu")
        hints.add("marino")
    return sorted(h for h in hints if h)


def _build_book_token_df(rows: List[Tuple[Dict, str]]) -> Dict[str, Counter]:
    by_book: Dict[str, Counter] = defaultdict(Counter)
    for metadata, text in rows:
        book_id = str(metadata.get("book_id", "")).strip()
        if not book_id:
            continue
        tokens = set(_tokenize_words(text[:2000]))
        for tok in tokens:
            if tok in STOPWORDS or tok in NOISE_TERMS:
                continue
            by_book[book_id][tok] += 1
    return by_book


def build_candidates(rows: Dict) -> Dict[str, List[Candidate]]:
    by_book: Dict[str, List[Candidate]] = defaultdict(list)
    metadatas = rows.get("metadatas") or []
    documents = rows.get("documents") or []

    pairs: List[Tuple[Dict, str]] = []
    for metadata, text in zip(metadatas, documents):
        cleaned = _clean_text(text)
        if len(cleaned) >= 220:
            pairs.append((metadata, cleaned))

    token_df_by_book = _build_book_token_df(pairs)

    seen_topics_by_book: Dict[str, set] = defaultdict(set)

    for metadata, text in pairs:
        if _is_noisy_chunk(text):
            continue

        book_id = str(metadata.get("book_id", "")).strip()
        if not book_id:
            continue
        book_name = str(metadata.get("book_name", book_id)).strip()

        hierarchy = _parse_hierarchy(metadata.get("hierarchy"))
        topic = _extract_topic(hierarchy, text)
        if not topic:
            continue
        if _is_bad_topic(topic):
            continue

        topic_key = topic.lower()
        if topic_key in seen_topics_by_book[book_id]:
            continue
        seen_topics_by_book[book_id].add(topic_key)

        expected_terms = _extract_terms(topic, token_df_by_book[book_id])
        if len(expected_terms) < 2:
            continue

        template = QUESTION_TEMPLATES[len(by_book[book_id]) % len(QUESTION_TEMPLATES)]
        query = template.format(topic=topic)

        by_book[book_id].append(
            Candidate(
                book_id=book_id,
                book_name=book_name,
                topic=topic,
                query=query,
                expected_terms=expected_terms,
                has_diagram=_has_diagram(metadata.get("diagram_paths")),
            )
        )

    return by_book


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    client = chromadb.Client(Settings(persist_directory=args.persist_directory, is_persistent=True))
    col = client.get_collection(args.collection_name)
    rows = col.get(include=["metadatas", "documents"])

    by_book = build_candidates(rows)
    if not by_book:
        print("No candidates generated from collection metadata.")
        return 1

    all_cases = []

    for book_id, candidates in sorted(by_book.items()):
        random.shuffle(candidates)
        selected = candidates[: args.per_book]
        source_any = _source_hints(book_id, selected[0].book_name if selected else book_id)

        diagram_required_quota = args.diagram_cases_miller if "miller" in book_id.lower() else 0
        diagram_selected = 0

        for idx, cand in enumerate(selected, start=1):
            require_diagram = False
            if cand.has_diagram and diagram_selected < diagram_required_quota:
                require_diagram = True
                diagram_selected += 1

            case = {
                "case_id": f"{book_id}_q{idx:03d}",
                "query": cand.query,
                "book_id": book_id,
                "expect_refused": False,
                "expected_terms": cand.expected_terms,
                "expected_source_any": source_any,
                "expected_answer_terms": [],
                "forbidden_answer_terms": [],
                "require_diagram": require_diagram,
            }
            all_cases.append(case)

        print(
            f"book={book_id} candidates={len(candidates)} selected={len(selected)} diagram_required={diagram_selected}"
        )

    payload = {
        "meta": {
            "generated_by": "evals/generate_bulk_cases.py",
            "seed": args.seed,
            "per_book": args.per_book,
            "diagram_cases_miller": args.diagram_cases_miller,
            "total_cases": len(all_cases),
        },
        "cases": all_cases,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(all_cases)} cases -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
