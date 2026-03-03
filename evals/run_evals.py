import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from llm.llm_service import LLMService
from llm.query_rewriter import QueryRewriter
from retrieval.retriever import Retriever


ABSTAIN_TEXT = "The answer is not available in the indexed medical sources."


@dataclass
class EvalCase:
    case_id: str
    query: str
    book_id: Optional[str]
    expect_refused: bool
    expected_terms: List[str]
    expected_source_any: List[str]
    expected_answer_terms: List[str]
    forbidden_answer_terms: List[str]
    require_diagram: bool


class EvalRunner:
    def __init__(
        self,
        retriever: Retriever,
        rewriter: QueryRewriter,
        llm_service: Optional[LLMService],
        include_llm: bool,
    ):
        self.retriever = retriever
        self.rewriter = rewriter
        self.llm_service = llm_service
        self.include_llm = include_llm

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        return term.lower() in text.lower()

    @staticmethod
    def _collect_source_text(results: List[Dict]) -> str:
        parts = []
        for item in results:
            metadata = item.get("metadata", {})
            hierarchy = metadata.get("hierarchy")
            if isinstance(hierarchy, list):
                hierarchy_text = " -> ".join(str(x) for x in hierarchy)
            elif hierarchy is None:
                hierarchy_text = ""
            else:
                hierarchy_text = str(hierarchy)
            parts.append(
                " ".join(
                    [
                        str(metadata.get("book_name", "")),
                        hierarchy_text,
                        str(item.get("text", "")),
                    ]
                )
            )
        return "\n".join(parts)

    def _evaluate_retrieval(self, case: EvalCase, retrieval_response: Dict) -> Dict:
        checks = []
        status = retrieval_response.get("status")

        checks.append(
            {
                "name": "status_matches_expectation",
                "passed": (status == "refused") == case.expect_refused,
                "details": {
                    "expected_refused": case.expect_refused,
                    "actual_status": status,
                },
            }
        )

        if status == "success":
            results = retrieval_response.get("results", [])
            source_text = self._collect_source_text(results)
            diagram_paths = self._extract_diagram_paths(results)

            for term in case.expected_terms:
                checks.append(
                    {
                        "name": f"retrieval_contains_term:{term}",
                        "passed": self._contains_term(source_text, term),
                    }
                )

            if case.expected_source_any:
                checks.append(
                    {
                        "name": "retrieval_contains_any_expected_source",
                        "passed": any(
                            self._contains_term(source_text, source)
                            for source in case.expected_source_any
                        ),
                        "details": {
                            "expected_any": case.expected_source_any,
                        },
                    }
                )

            if case.require_diagram:
                checks.append(
                    {
                        "name": "retrieval_contains_diagram_paths",
                        "passed": len(diagram_paths) > 0,
                        "details": {
                            "diagram_paths_count": len(diagram_paths),
                        },
                    }
                )
                checks.append(
                    {
                        "name": "retrieved_diagram_file_exists",
                        "passed": any(os.path.exists(path) for path in diagram_paths),
                        "details": {
                            "existing_count": sum(1 for path in diagram_paths if os.path.exists(path)),
                        },
                    }
                )

        return {
            "checks": checks,
            "passed": all(c["passed"] for c in checks),
        }

    def _evaluate_answer(self, case: EvalCase, retrieval_response: Dict, answer_text: str) -> Dict:
        checks = []
        normalized_answer = self._normalize(answer_text)

        if case.expect_refused:
            checks.append(
                {
                    "name": "refusal_response_contains_abstain_text",
                    "passed": self._contains_term(normalized_answer, ABSTAIN_TEXT.lower()),
                }
            )
        else:
            for term in case.expected_answer_terms:
                checks.append(
                    {
                        "name": f"answer_contains_term:{term}",
                        "passed": self._contains_term(normalized_answer, term),
                    }
                )

            for term in case.forbidden_answer_terms:
                checks.append(
                    {
                        "name": f"answer_excludes_term:{term}",
                        "passed": not self._contains_term(normalized_answer, term),
                    }
                )

            if retrieval_response.get("status") == "success":
                checks.append(
                    {
                        "name": "answer_not_empty",
                        "passed": len(normalized_answer) > 20,
                    }
                )

        return {
            "checks": checks,
            "passed": all(c["passed"] for c in checks),
        }

    def run_case(self, case: EvalCase) -> Dict:
        rewritten_query = self.rewriter.rewrite(case.query, history=[])
        retrieval_response = self.retriever.retrieve(rewritten_query, book_id=case.book_id)
        retrieved_diagram_paths = self._extract_diagram_paths(
            retrieval_response.get("results", []) if retrieval_response.get("status") == "success" else []
        )

        retrieval_eval = self._evaluate_retrieval(case, retrieval_response)
        ranking_metrics = self._compute_ranking_metrics(case, retrieval_response)

        answer_text = None
        answer_eval = None

        if self.include_llm:
            if self.llm_service is None:
                raise RuntimeError("LLM evaluation requested but llm_service is None")

            answer_text = "".join(
                self.llm_service.generate_answer_stream(rewritten_query, retrieval_response)
            ).strip()
            answer_eval = self._evaluate_answer(case, retrieval_response, answer_text)

        case_passed = retrieval_eval["passed"] and (answer_eval["passed"] if answer_eval else True)

        return {
            "case_id": case.case_id,
            "query": case.query,
            "rewritten_query": rewritten_query,
            "retrieval": retrieval_response,
            "retrieved_diagram_paths": retrieved_diagram_paths,
            "retrieval_eval": retrieval_eval,
            "ranking_metrics": ranking_metrics,
            "answer": answer_text,
            "answer_eval": answer_eval,
            "passed": case_passed,
        }

    @staticmethod
    def _extract_diagram_paths(results: List[Dict]) -> List[str]:
        paths: List[str] = []
        for item in results:
            metadata = item.get("metadata", {})
            raw = metadata.get("diagram_paths")
            parsed: List[str] = []
            if isinstance(raw, list):
                parsed = [p for p in raw if isinstance(p, str) and p.strip()]
            elif isinstance(raw, str):
                candidate = raw.strip()
                if candidate:
                    try:
                        arr = json.loads(candidate)
                        if isinstance(arr, list):
                            parsed = [p for p in arr if isinstance(p, str) and p.strip()]
                        else:
                            parsed = [candidate]
                    except Exception:
                        parsed = [candidate]
            for path in parsed:
                if path not in paths:
                    paths.append(path)
        return paths

    def _compute_ranking_metrics(self, case: EvalCase, retrieval_response: Dict) -> Optional[Dict]:
        evaluable = (
            not case.expect_refused
            and (bool(case.expected_terms) or bool(case.expected_source_any) or case.require_diagram)
        )
        if not evaluable:
            return None

        results = retrieval_response.get("results", []) if retrieval_response.get("status") == "success" else []
        if not results:
            return {
                "evaluable": True,
                "k": 0,
                "mrr": 0.0,
                "dcg": 0.0,
                "idcg": 0.0,
                "ndcg": 0.0,
                "hit_rate": 0.0,
                "precision_at_k": 0.0,
                "term_recall": 0.0 if case.expected_terms else None,
                "source_hit": 0.0 if case.expected_source_any else None,
                "relevant_count": 0,
            }

        graded_rels: List[int] = []
        binary_rels: List[int] = []
        matched_terms_all = set()
        source_hit = False

        for item in results:
            item_text = self._collect_source_text([item])

            matched_terms = {term for term in case.expected_terms if self._contains_term(item_text, term)}
            matched_terms_all.update(matched_terms)

            has_expected_source = (
                any(self._contains_term(item_text, source) for source in case.expected_source_any)
                if case.expected_source_any
                else False
            )
            if has_expected_source:
                source_hit = True

            has_diagram = len(self._extract_diagram_paths([item])) > 0

            # Graded relevance:
            # +2 max for expected terms, +1 for expected source match, +1 for diagram match when required.
            grade = min(2, len(matched_terms))
            if has_expected_source:
                grade += 1
            if case.require_diagram and has_diagram:
                grade += 1

            graded_rels.append(grade)
            binary_rels.append(1 if grade > 0 else 0)

        first_relevant_rank = next((idx + 1 for idx, rel in enumerate(binary_rels) if rel > 0), None)
        mrr = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0

        dcg = sum(((2 ** rel) - 1) / math.log2(idx + 2) for idx, rel in enumerate(graded_rels))
        ideal_rels = sorted(graded_rels, reverse=True)
        idcg = sum(((2 ** rel) - 1) / math.log2(idx + 2) for idx, rel in enumerate(ideal_rels))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        hit_rate = 1.0 if any(binary_rels) else 0.0
        precision_at_k = sum(binary_rels) / len(binary_rels)
        term_recall = (
            len(matched_terms_all) / len(set(case.expected_terms))
            if case.expected_terms
            else None
        )
        source_hit_value = 1.0 if source_hit else 0.0 if case.expected_source_any else None

        return {
            "evaluable": True,
            "k": len(results),
            "mrr": round(mrr, 6),
            "dcg": round(dcg, 6),
            "idcg": round(idcg, 6),
            "ndcg": round(ndcg, 6),
            "hit_rate": round(hit_rate, 6),
            "precision_at_k": round(precision_at_k, 6),
            "term_recall": round(term_recall, 6) if term_recall is not None else None,
            "source_hit": source_hit_value,
            "relevant_count": int(sum(binary_rels)),
        }


def _load_cases(path: str) -> List[EvalCase]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_cases = payload.get("cases", [])
    cases: List[EvalCase] = []

    for row in raw_cases:
        cases.append(
            EvalCase(
                case_id=row["case_id"],
                query=row["query"],
                book_id=row.get("book_id"),
                expect_refused=bool(row.get("expect_refused", False)),
                expected_terms=row.get("expected_terms", []),
                expected_source_any=row.get("expected_source_any", []),
                expected_answer_terms=row.get("expected_answer_terms", []),
                forbidden_answer_terms=row.get("forbidden_answer_terms", []),
                require_diagram=bool(row.get("require_diagram", False)),
            )
        )

    return cases


def _write_report(output_dir: str, report: Dict) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"eval_report_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path


def _summarize(results: List[Dict]) -> Tuple[int, int]:
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    return passed, total


def _aggregate_ranking_metrics(results: List[Dict]) -> Dict:
    ranking_rows = [row for row in (r.get("ranking_metrics") for r in results) if row and row.get("evaluable")]
    if not ranking_rows:
        return {
            "ranking_cases": 0,
            "mean_mrr": None,
            "mean_dcg": None,
            "mean_ndcg": None,
            "mean_hit_rate": None,
            "mean_precision_at_k": None,
            "mean_term_recall": None,
            "mean_source_hit": None,
        }

    def _mean(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return round(sum(values) / len(values), 6)

    term_recalls = [row["term_recall"] for row in ranking_rows if row.get("term_recall") is not None]
    source_hits = [row["source_hit"] for row in ranking_rows if row.get("source_hit") is not None]

    return {
        "ranking_cases": len(ranking_rows),
        "mean_mrr": _mean([row["mrr"] for row in ranking_rows]),
        "mean_dcg": _mean([row["dcg"] for row in ranking_rows]),
        "mean_ndcg": _mean([row["ndcg"] for row in ranking_rows]),
        "mean_hit_rate": _mean([row["hit_rate"] for row in ranking_rows]),
        "mean_precision_at_k": _mean([row["precision_at_k"] for row in ranking_rows]),
        "mean_term_recall": _mean(term_recalls),
        "mean_source_hit": _mean(source_hits),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AIRA evals")
    parser.add_argument("--cases", default="evals/cases.json", help="Path to eval cases JSON")
    parser.add_argument(
        "--output-dir",
        default="evals/reports",
        help="Directory where eval report JSON is written",
    )
    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="Also generate answers and run answer-level checks",
    )
    parser.add_argument("--model", default="mistral", help="Ollama model for rewrite/generation")
    parser.add_argument("--top-k", type=int, default=8, help="Retriever top_k")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.35,
        help="Retriever similarity threshold",
    )
    parser.add_argument(
        "--min-support-chunks",
        type=int,
        default=1,
        help="Minimum number of retrieved chunks required before generation",
    )
    parser.add_argument(
        "--min-avg-similarity",
        type=float,
        default=0.0,
        help="Minimum average similarity across retrieved chunks required before generation",
    )
    parser.add_argument(
        "--collection-name",
        default="medical_knowledge",
        help="Chroma collection name",
    )
    parser.add_argument(
        "--persist-directory",
        default="vectorstore",
        help="Chroma persist directory",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="If > 0, run only the first N cases",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cases = _load_cases(args.cases)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    if not cases:
        print("No eval cases found.")
        return 1

    try:
        retriever = Retriever(
            persist_directory=args.persist_directory,
            collection_name=args.collection_name,
            similarity_threshold=args.similarity_threshold,
            top_k=args.top_k,
            min_support_chunks=args.min_support_chunks,
            min_avg_similarity=args.min_avg_similarity,
        )
        rewriter = QueryRewriter(model=args.model)
        llm_service = LLMService(model=args.model) if args.include_llm else None
    except Exception as exc:
        print(f"Failed to initialize eval dependencies: {exc}")
        return 1

    runner = EvalRunner(
        retriever=retriever,
        rewriter=rewriter,
        llm_service=llm_service,
        include_llm=args.include_llm,
    )

    results = []
    for case in cases:
        try:
            result = runner.run_case(case)
        except Exception as exc:
            result = {
                "case_id": case.case_id,
                "query": case.query,
                "error": str(exc),
                "passed": False,
            }
        results.append(result)
        print(f"[{ 'PASS' if result['passed'] else 'FAIL' }] {case.case_id}")

    passed, total = _summarize(results)
    ranking_summary = _aggregate_ranking_metrics(results)

    report = {
        "generated_at": datetime.now().isoformat(),
        "include_llm": args.include_llm,
        "model": args.model,
        "summary": {
            "passed": passed,
            "total": total,
            "pass_rate": (passed / total) if total else 0.0,
            "diagram_cases_total": sum(1 for c in cases if c.require_diagram),
            "diagram_cases_passed": sum(
                1
                for c, r in zip(cases, results)
                if c.require_diagram and r.get("passed")
            ),
            "ranking": ranking_summary,
        },
        "results": results,
    }

    out_path = _write_report(args.output_dir, report)

    print("\nEval Summary")
    print("------------")
    print(f"Passed: {passed}/{total}")
    print(f"Pass rate: {(passed / total) * 100:.1f}%")
    if ranking_summary.get("ranking_cases", 0) > 0:
        print("Ranking metrics:")
        print(f"  cases: {ranking_summary['ranking_cases']}")
        print(f"  mean_mrr: {ranking_summary['mean_mrr']}")
        print(f"  mean_dcg: {ranking_summary['mean_dcg']}")
        print(f"  mean_ndcg: {ranking_summary['mean_ndcg']}")
        print(f"  mean_hit_rate: {ranking_summary['mean_hit_rate']}")
        print(f"  mean_precision_at_k: {ranking_summary['mean_precision_at_k']}")
        if ranking_summary.get("mean_term_recall") is not None:
            print(f"  mean_term_recall: {ranking_summary['mean_term_recall']}")
        if ranking_summary.get("mean_source_hit") is not None:
            print(f"  mean_source_hit: {ranking_summary['mean_source_hit']}")
    print(f"Report: {out_path}")

    return 0 if passed == total else 2


if __name__ == "__main__":
    raise SystemExit(main())
