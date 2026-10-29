"""Small faithfulness-style evaluation.

For each question in ``eval_set.json`` this script asks the assistant
(offline fallback mode unless a key is configured) and scores two
things:

1. **Keyword recall** — fraction of expected keywords present in the
   answer (a rough relevance signal, not a quality judgement).
2. **Grounding** — whether the answer shares content words with the
   retrieved source chunks (the same check the critic performs).

No metric here is a production claim: this is a tiny sample set for a
portfolio project, and the scores are reported, not marketed. Exits
non-zero if the average keyword recall is below 0.5.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_rag_assistant.embeddings import tokenize  # noqa: E402
from agentic_rag_assistant.service import AssistantService  # noqa: E402

_STOPWORDS = {
    "the", "a", "an", "is", "are", "what", "how", "of", "to", "in",
    "on", "for", "and", "with", "by",
}


def _content_words(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 2}


def main() -> int:
    """Run the eval set and print per-item + aggregate scores."""
    root = Path(__file__).resolve().parents[1]
    eval_set = json.loads((root / "evals" / "eval_set.json").read_text())
    service = AssistantService()
    service.load_sample_docs()

    keyword_scores: list[float] = []
    grounded = 0
    for item in eval_set:
        result = service.ask(item["question"])
        answer = result["answer"].lower()
        hits = sum(1 for kw in item["expected_keywords"] if kw in answer)
        score = hits / len(item["expected_keywords"])
        keyword_scores.append(score)
        is_grounded = result["critic_passed"]
        grounded += int(is_grounded)
        print(f"Q: {item['question']}")
        print(f"  keyword recall: {score:.2f} ({hits}/{len(item['expected_keywords'])})")
        print(f"  grounded (critic): {is_grounded}")
        print(f"  sources: {result['sources']}")

    avg = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0.0
    print(f"\nAverage keyword recall: {avg:.2f}")
    print(f"Grounded answers: {grounded}/{len(eval_set)}")
    return 0 if avg >= 0.5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
