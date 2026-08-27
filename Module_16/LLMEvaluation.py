from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# 1. TEXT-GEN METRICS
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def exact_match(candidate: str, reference: str) -> float:
    """Binary metric -- best for closed-form, single-correct-answer tasks."""
    return 1.0 if candidate.strip().lower() == reference.strip().lower() else 0.0


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu_n(candidate: str, reference: str, n: int = 4) -> float:
    """
    Simplified BLEU: n-gram precision (up to n) with a brevity penalty.
    Real BLEU uses a geometric mean across 1..n gram precisions -- this
    keeps that shape but skips smoothing details for clarity.
    """
    cand_tokens, ref_tokens = _tokenize(candidate), _tokenize(reference)
    if not cand_tokens:
        return 0.0
    precisions = []
    for k in range(1, n + 1):
        cand_ngrams, ref_ngrams = _ngrams(cand_tokens, k), _ngrams(ref_tokens, k)
        if not cand_ngrams:
            continue
        overlap = sum(min(count, ref_ngrams.get(gram, 0)) for gram, count in cand_ngrams.items())
        total = sum(cand_ngrams.values())
        precisions.append(overlap / total if total else 0.0)
    if not precisions or any(p == 0 for p in precisions):
        geo_mean = 0.0
    else:
        geo_mean = statistics.geometric_mean(precisions)
    brevity_penalty = min(1.0, len(cand_tokens) / max(len(ref_tokens), 1))
    return geo_mean * brevity_penalty

def rouge_n(candidate: str, reference: str, n: int = 1) -> float:
    """Recall-oriented n-gram overlap (ROUGE-1 / ROUGE-2 depending on n)."""
    cand_tokens, ref_tokens = _tokenize(candidate), _tokenize(reference)
    ref_ngrams = _ngrams(ref_tokens, n)
    cand_ngrams = _ngrams(cand_tokens, n)
    if not ref_ngrams:
        return 0.0
    overlap = sum(min(count, cand_ngrams.get(gram, 0)) for gram, count in ref_ngrams.items())
    return overlap / sum(ref_ngrams.values())

def rouge_l(candidate: str, reference: str) -> float:
    """Longest common subsequence based F1 -- rewards in-order overlap without requiring contiguity."""
    a, b = _tokenize(candidate), _tokenize(reference)
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[len(a)][len(b)]
    precision, recall = lcs / len(a), lcs / len(b)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

def embedding_similarity(candidate: str, reference: str, dims: int = 64, n: int = 3) -> float:
    """
    Stand-in for BERTScore: character n-gram hashed vectors + cosine
    similarity, so paraphrases without lexical overlap still score close.
    Swap for a real contextual embedding model in production.
    """
    import hashlib
    import math

    def embed(text: str) -> list[float]:
        vec = [0.0] * dims
        cleaned = re.sub(r"[^a-z0-9 ]", "", text.lower())
        grams = [cleaned[i:i + n] for i in range(len(cleaned) - n + 1)] or [cleaned]
        for g in grams:
            idx = int(hashlib.md5(g.encode()).hexdigest(), 16) % dims
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    va, vb = embed(candidate), embed(reference)
    return sum(x * y for x, y in zip(va, vb))

# ---------------------------------------------------------------------------
# 2. LLM-AS-A-JUDGE
# ---------------------------------------------------------------------------

@dataclass
class JudgeVerdict:
    score: Optional[float] = None          # pointwise: 1-5
    winner: Optional[str] = None           # pairwise: "A", "B", or "tie"
    reasoning: str = ""

class JudgeModel:
    """
    Stand-in for a strong LLM used as a judge. Scores deterministically off
    reference/context overlap so the demo is reproducible, but the *shape*
    of the interface (rubric-based prompt, chain-of-thought reasoning,
    structured output) mirrors a real judge prompt.
    """

    def judge_pointwise(self, response: str, reference: str, rubric: str = "accuracy") -> JudgeVerdict:
        overlap = embedding_similarity(response, reference)
        score = round(1 + overlap * 4, 1)  # map similarity [0,1] -> score [1,5]
        return JudgeVerdict(
            score=score,
            reasoning=f"Response semantic overlap with reference: {overlap:.2f} (rubric: {rubric})",
        )

    def _judge_pair_once(self, response_a: str, response_b: str, reference: str) -> str:
        score_a = embedding_similarity(response_a, reference)
        score_b = embedding_similarity(response_b, reference)
        if abs(score_a - score_b) < 0.03:
            return "tie"
        return "A" if score_a > score_b else "B"

    def judge_pairwise(self, response_a: str, response_b: str, reference: str) -> JudgeVerdict:
        """
        Pairwise comparison with position-bias mitigation: judge both
        orderings (A,B) and (B,A) and only trust the verdict if it's
        consistent across the swap -- otherwise report a tie/uncertain
        rather than silently picking whichever came first.
        """
        first_pass = self._judge_pair_once(response_a, response_b, reference)
        swapped = self._judge_pair_once(response_b, response_a, reference)
        swapped_normalized = {"A": "B", "B": "A", "tie": "tie"}[swapped]

        if first_pass == swapped_normalized:
            winner = first_pass
            note = "consistent across position swap"
        else:
            winner = "tie"
            note = f"position-bias check failed (got {first_pass} then {swapped_normalized}) -- treating as tie"

        return JudgeVerdict(winner=winner, reasoning=note)

# ---------------------------------------------------------------------------
# 3. HALLUCINATION DETECTION
# ---------------------------------------------------------------------------

class SelfCheckConsistency:
    """
    SelfCheckGPT-style: sample the generator N times for the same prompt,
    then measure pairwise disagreement across samples. High divergence
    suggests the model isn't anchored to real knowledge on this prompt.
    """

    def check(self, samples: list[str]) -> tuple[float, str]:
        if len(samples) < 2:
            return 0.0, "not enough samples to check consistency"
        pairwise_scores = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                pairwise_scores.append(embedding_similarity(samples[i], samples[j]))
        avg_similarity = statistics.mean(pairwise_scores)
        inconsistency_score = 1 - avg_similarity  # higher = more likely hallucinated
        verdict = "likely hallucination (samples diverge)" if inconsistency_score > 0.5 else "consistent across samples"
        return round(inconsistency_score, 2), verdict

class ClaimFactChecker:
    """
    FactScore-style: decompose a response into atomic claims (naively, by
    sentence), then check each against the provided context, reporting the
    proportion supported rather than a single whole-response verdict.
    """

    def decompose(self, response: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", response.strip())
        return [s for s in sentences if s]

    def score(self, response: str, context: str) -> tuple[float, list[dict]]:
        claims = self.decompose(response)
        context_terms = set(_tokenize(context))
        results = []
        supported_count = 0
        for claim in claims:
            claim_terms = set(_tokenize(claim))
            content_terms = claim_terms - {"the", "a", "an", "is", "was", "in", "of", "to", "and"}
            if not content_terms:
                supported = True
            else:
                overlap_ratio = len(content_terms & context_terms) / len(content_terms)
                supported = overlap_ratio >= 0.5
            results.append({"claim": claim, "supported": supported})
            supported_count += int(supported)
        fact_score = supported_count / len(claims) if claims else 1.0
        return round(fact_score, 2), results

# ---------------------------------------------------------------------------
# 4. EVAL DATASET (golden set)
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    id: str
    input: str
    reference: str
    context: str = ""
    category: str = "standard"  # e.g. "standard", "adversarial", "out_of_scope"

@dataclass
class EvalDataset:
    version: str
    cases: list[EvalCase] = field(default_factory=list)

# ---------------------------------------------------------------------------
# MOCK GENERATOR (system under test)
# ---------------------------------------------------------------------------

class MockGenerator:
    """The pipeline being evaluated. Deterministic-ish canned outputs for the demo."""

    RESPONSES = {
        "refund": "We offer a full refund within 30 days of purchase, no questions asked.",
        "revenue": "Revenue grew 42% in Q3, driven by the EMEA region and new enterprise deals.",
        "capital": "The capital of France is Paris.",
        "hallucinate": "Our product was founded in 1850 and has won the Nobel Prize for customer service.",
    }

    def generate(self, prompt: str, variant: int = 0) -> str:
        lowered = prompt.lower()
        for key, response in self.RESPONSES.items():
            if key in lowered:
                if variant == 0:
                    return response
                # Simulate mild wording drift across samples for the consistency check
                return response.replace(".", "") + ("." if variant % 2 == 0 else " indeed.")
        return "I don't have information about that."

# ---------------------------------------------------------------------------
# 5. EVAL RUNNER + CI-STYLE REGRESSION GATE
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case_id: str
    scores: dict

@dataclass
class SuiteResult:
    dataset_version: str
    case_results: list[CaseResult]
    aggregate: dict

class EvalRunner:
    def __init__(self, generator: MockGenerator, judge: JudgeModel):
        self.generator = generator
        self.judge = judge
        self.fact_checker = ClaimFactChecker()
        self.self_check = SelfCheckConsistency()
        self.baseline: Optional[SuiteResult] = None
        self.trend_log: list[dict] = []

    def run_case(self, case: EvalCase) -> CaseResult:
        response = self.generator.generate(case.input)

        scores = {
            "exact_match": exact_match(response, case.reference),
            "rouge_l": round(rouge_l(response, case.reference), 2),
            "bertscore_proxy": round(embedding_similarity(response, case.reference), 2),
            "judge_pointwise": self.judge.judge_pointwise(response, case.reference).score,
        }

        if case.context:
            fact_score, claim_results = self.fact_checker.score(response, case.context)
            scores["fact_score"] = fact_score
            unsupported = [c["claim"] for c in claim_results if not c["supported"]]
            if unsupported:
                scores["unsupported_claims"] = unsupported

        if case.category == "adversarial":
            samples = [self.generator.generate(case.input, variant=v) for v in range(3)]
            inconsistency, verdict = self.self_check.check(samples)
            scores["selfcheck_inconsistency"] = inconsistency
            scores["selfcheck_verdict"] = verdict

        return CaseResult(case_id=case.id, scores=scores)

    def run_suite(self, dataset: EvalDataset) -> SuiteResult:
        case_results = [self.run_case(c) for c in dataset.cases]
        aggregate = {
            "avg_exact_match": round(statistics.mean(r.scores["exact_match"] for r in case_results), 3),
            "avg_rouge_l": round(statistics.mean(r.scores["rouge_l"] for r in case_results), 3),
            "avg_bertscore_proxy": round(statistics.mean(r.scores["bertscore_proxy"] for r in case_results), 3),
            "avg_judge_score": round(statistics.mean(r.scores["judge_pointwise"] for r in case_results), 3),
        }
        return SuiteResult(dataset_version=dataset.version, case_results=case_results, aggregate=aggregate)

    def gate(self, result: SuiteResult, thresholds: dict, max_regression: float = 0.05) -> tuple[bool, list[str]]:
        """
        CI-style pass/fail gate: absolute thresholds + regression tolerance
        against the stored baseline. Records the run in the trend log
        regardless of pass/fail, so drift is visible even across passing runs.
        """
        failures = []

        for metric, min_value in thresholds.items():
            actual = result.aggregate.get(metric)
            if actual is None:
                continue
            if actual < min_value:
                failures.append(f"{metric}={actual} below threshold {min_value}")

        if self.baseline is not None:
            for metric, baseline_value in self.baseline.aggregate.items():
                current_value = result.aggregate.get(metric)
                if current_value is None:
                    continue
                drop = baseline_value - current_value
                if drop > max_regression:
                    failures.append(
                        f"{metric} regressed {drop:.3f} vs baseline "
                        f"({baseline_value} -> {current_value}, tolerance {max_regression})"
                    )

        passed = len(failures) == 0
        self.trend_log.append({"dataset_version": result.dataset_version, **result.aggregate, "passed": passed})
        return passed, failures

    def set_baseline(self, result: SuiteResult):
        self.baseline = result

# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generator = MockGenerator()
    judge = JudgeModel()
    runner = EvalRunner(generator, judge)

    dataset_v1 = EvalDataset(
        version="v1",
        cases=[
            EvalCase("case_refund", "What is your refund policy?",
                     reference="We offer a full refund within 30 days of purchase.",
                     category="standard"),
            EvalCase("case_revenue", "What was our Q3 revenue growth?",
                     reference="Revenue grew 42% in Q3, driven by EMEA.",
                     context="Q3 report: Revenue grew 42% in the EMEA region on new enterprise deals.",
                     category="standard"),
            EvalCase("case_capital", "What is the capital of France?",
                     reference="Paris",
                     category="standard"),
            EvalCase("case_adversarial", "Tell me about our hallucinate product history",
                     reference="Founded in 2015, focused on customer service tooling.",
                     context="The product launched in 2015 as an internal support tool.",
                     category="adversarial"),
        ],
    )

    print("=== Run 1: establishing baseline (dataset v1) ===\n")
    result_v1 = runner.run_suite(dataset_v1)
    for r in result_v1.case_results:
        print(f"[{r.case_id}] {r.scores}")
    print(f"\nAggregate: {result_v1.aggregate}")

    thresholds = {"avg_bertscore_proxy": 0.5, "avg_judge_score": 3.0}
    passed, failures = runner.gate(result_v1, thresholds)
    print(f"\nGate result: {'PASS' if passed else 'FAIL'}")
    if failures:
        for f in failures:
            print(f"  - {f}")
    runner.set_baseline(result_v1)

    # --- Simulate a "regression" run: swap in a generator that's worse on one case ---
    print("\n\n=== Run 2: simulating a regression (e.g. after a prompt change) ===\n")

    class RegressedGenerator(MockGenerator):
        def generate(self, prompt: str, variant: int = 0) -> str:
            if "refund" in prompt.lower():
                return "Not sure, check the website."  # degraded response
            return super().generate(prompt, variant)

    runner_v2 = EvalRunner(RegressedGenerator(), judge)
    runner_v2.baseline = result_v1  # compare against the same baseline
    result_v2 = runner_v2.run_suite(dataset_v1)
    for r in result_v2.case_results:
        print(f"[{r.case_id}] {r.scores}")
    print(f"\nAggregate: {result_v2.aggregate}")

    passed_v2, failures_v2 = runner_v2.gate(result_v2, thresholds)
    print(f"\nGate result: {'PASS' if passed_v2 else 'FAIL'}")
    for f in failures_v2:
        print(f"  - {f}")

    print("\n\n=== Trend log (what a CI dashboard would track over time) ===")
    for entry in runner_v2.trend_log:
        print(entry)