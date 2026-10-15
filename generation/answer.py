"""Grounded answering: retrieve -> prompt -> LLM -> citation check -> disclaimer.

    python -m generation.answer "What is the punishment for cheating under BNS?"
"""
import argparse
from dataclasses import dataclass, field
from typing import Optional, Sequence

from config import setup_logging, settings
from retrieval.search import Hit

from .citations import check_citations
from .llm import LLM
from .prompt import MESSAGES, NOT_FOUND, detect_language, system_prompt, user_prompt


@dataclass
class Answer:
    text: str
    lang: str
    grounded: bool                                   # False when the answer is the "not in sources" reply
    citations: list[str] = field(default_factory=list)
    sources: list[Hit] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    disclaimer: str = ""


def _refuse(lang: str, dropped: Optional[list[str]] = None) -> Answer:
    m = MESSAGES[lang]
    return Answer(m["not_found"], lang, False, [], [], dropped or [], m["disclaimer"])


def generate_answer(question: str, hits: Sequence[Hit], llm: LLM, lang: Optional[str] = None) -> Answer:
    """Answer from the given hits only. Unsupported sentences are removed; nothing left -> plain refusal."""
    lang = lang or detect_language(question)
    if not hits:
        return _refuse(lang)
    raw = llm.complete(system_prompt(lang), user_prompt(question, hits)).strip()
    if not raw or NOT_FOUND in raw:
        return _refuse(lang)
    checked = check_citations(raw, [h.cite for h in hits])
    if not checked.text:
        return _refuse(lang, checked.dropped)
    used = set(checked.citations)
    return Answer(checked.text, lang, True, checked.citations, [h for h in hits if h.cite in used],
                  checked.dropped, MESSAGES[lang]["disclaimer"])


def answer_question(question: str, llm: Optional[LLM] = None, k: Optional[int] = None, lang: Optional[str] = None,
                    **filters) -> Answer:
    """Full pipeline: hybrid retrieval, then generate_answer."""
    from retrieval.hybrid import hybrid_search  # heavy imports kept out of module load
    from .llm import make_llm
    hits = hybrid_search(question, k or settings.answer_top_k, **filters)
    return generate_answer(question, hits, llm or make_llm(), lang)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question")
    ap.add_argument("--k", type=int)
    ap.add_argument("--lang", choices=("en", "hi"))
    a = ap.parse_args(argv)
    setup_logging()
    ans = answer_question(a.question, k=a.k, lang=a.lang)
    print(ans.text)
    if ans.citations:
        print("\nSources: " + "; ".join(ans.citations))
    print(f"\n{ans.disclaimer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
