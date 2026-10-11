"""Minimal evaluation harness: run a model function over prompts and score the outputs."""

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass
class EvalResult:
    prompt: str
    output: str
    score: float


def run_eval(
    model: Callable[[str], str],
    prompts: Iterable[str],
    scorer: Callable[[str, str], float],
) -> list[EvalResult]:
    """Run `model` on each prompt and score each (prompt, output) pair."""
    results = []
    for prompt in prompts:
        output = model(prompt)
        results.append(EvalResult(prompt, output, scorer(prompt, output)))
    return results


def mean_score(results: list[EvalResult]) -> float:
    return sum(r.score for r in results) / len(results) if results else 0.0
