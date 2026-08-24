"""
Mini Project: LLM Cost-Optimization Gateway
=============================================

Covers, end to end, every concept from the "Caching and Cost Optimization"
module:

  1. Exact-Match Caching   -> ExactMatchCache   (hash + TTL, Redis-style)
  2. Semantic Caching      -> SemanticCache     (embeddings + threshold tuning)
  3. Prompt Compression    -> PromptCompressor  (extractive compression)
  4. Model Routing         -> ModelRouter       (complexity classify + cascade)
  5. LiteLLM-style gateway -> MockLLMProvider + LLMGateway
  6. Cost tracking/report  -> CostTracker

No external API calls / no network required -- providers and embeddings are
mocked so the whole pipeline runs offline and deterministically. In a real
system you'd swap MockLLMProvider for LiteLLM's `completion()`, and
`simple_embedding()` for a real embedding model (OpenAI, sentence-transformers,
etc.), and ExactMatchCache/SemanticCache's in-memory dicts for actual Redis /
Redis-Stack (vector) calls. The interfaces are shaped so that swap is a
drop-in change, not a redesign.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# 1. EXACT-MATCH CACHING
# ---------------------------------------------------------------------------

class ExactMatchCache:
    """
    Hash-keyed cache with TTL, standing in for Redis (SET ... EX <ttl>).

    The hash includes every parameter that affects the output -- not just
    the prompt text -- so two requests that differ only in model or
    temperature are correctly treated as different cache entries.
    """

    def __init__(self):
        self._store: dict[str, tuple[str, float]] = {}  # hash -> (response, expires_at)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(model: str, prompt: str, temperature: float) -> str:
        raw = f"{model}|{temperature}|{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, model: str, prompt: str, temperature: float) -> Optional[str]:
        key = self._key(model, prompt, temperature)
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        response, expires_at = entry
        if time.time() > expires_at:
            # TTL-based (passive) invalidation
            del self._store[key]
            self.misses += 1
            return None
        self.hits += 1
        return response

    def set(self, model: str, prompt: str, temperature: float, response: str, ttl_seconds: int = 3600):
        key = self._key(model, prompt, temperature)
        self._store[key] = (response, time.time() + ttl_seconds)

    def invalidate_all(self):
        """Manual/admin purge -- the emergency escape hatch."""
        self._store.clear()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ---------------------------------------------------------------------------
# 2. SEMANTIC CACHING
# ---------------------------------------------------------------------------

def simple_embedding(text: str, dims: int = 64, n: int = 3) -> list[float]:
    """
    Stand-in for a real embedding model (e.g. text-embedding-3, or a
    sentence-transformer). Uses the hashing trick over character n-grams
    (not whole words) so paraphrases -- even ones sharing few exact words,
    or with minor spelling/morphology differences ("refund" vs "refunds")
    -- still land close together in vector space. This is still a crude
    lexical proxy, not true semantic understanding (it won't catch synonyms
    like "affordable" vs "cheap" if the surface text doesn't overlap) --
    that gap is exactly why production systems use a real embedding model.
    """
    vec = [0.0] * dims
    cleaned = re.sub(r"[^a-z0-9 ]", "", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    grams = [cleaned[i:i + n] for i in range(len(cleaned) - n + 1)] or [cleaned]
    for g in grams:
        idx = int(hashlib.md5(g.encode()).hexdigest(), 16) % dims
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class SemanticCacheEntry:
    prompt: str
    embedding: list[float]
    response: str
    model: str


class SemanticCache:
    """
    Vector-similarity cache (stand-in for GPTCache / Redis Stack + vector
    search). Threshold is the key tuning knob: too loose -> false-positive
    wrong answers served confidently; too strict -> barely beats exact match.
    """

    def __init__(self, similarity_threshold: float = 0.5):
        self.threshold = similarity_threshold
        self._entries: list[SemanticCacheEntry] = []
        self.hits = 0
        self.misses = 0

    def get(self, model: str, prompt: str) -> Optional[tuple[str, float]]:
        query_vec = simple_embedding(prompt)
        best_score, best_entry = 0.0, None
        for entry in self._entries:
            if entry.model != model:
                continue
            score = cosine_similarity(query_vec, entry.embedding)
            if score > best_score:
                best_score, best_entry = score, entry

        if best_entry and best_score >= self.threshold:
            self.hits += 1
            return best_entry.response, best_score

        self.misses += 1
        return None

    def set(self, model: str, prompt: str, response: str):
        self._entries.append(
            SemanticCacheEntry(prompt, simple_embedding(prompt), response, model)
        )

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ---------------------------------------------------------------------------
# 3. PROMPT COMPRESSION
# ---------------------------------------------------------------------------

class PromptCompressor:
    """
    Extractive compression (LLMLingua-style): drop low-information tokens
    (filler words, stopwords) while preserving content words, numbers, and
    named entities. Real LLMLingua uses a small LM to score token
    importance/perplexity; this uses a stopword list as a cheap stand-in
    for the same idea, plus a length gate so short prompts aren't touched
    (compressing a short prompt isn't worth the overhead).
    """

    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "of", "to", "in", "on", "at", "for", "with", "and", "or", "but",
        "that", "this", "these", "those", "it", "its", "as", "by", "from",
        "please", "could", "you", "kindly", "just", "really", "very",
    }

    def __init__(self, min_length_to_compress: int = 40):
        self.min_length_to_compress = min_length_to_compress

    def compress(self, prompt: str) -> tuple[str, float]:
        """Returns (compressed_prompt, compression_ratio)."""
        if len(prompt) < self.min_length_to_compress:
            return prompt, 1.0  # too short to bother -- fixed overhead isn't worth it

        tokens = prompt.split()
        kept = [t for t in tokens if re.sub(r"[^a-z]", "", t.lower()) not in self.STOPWORDS]
        compressed = " ".join(kept) if kept else prompt
        ratio = len(prompt) / max(len(compressed), 1)
        return compressed, ratio


# ---------------------------------------------------------------------------
# 4. MODEL ROUTING (with cascading)
# ---------------------------------------------------------------------------

CHEAP_MODEL = "small-model"
STRONG_MODEL = "large-model"

# Simulated $ cost per 1K tokens (input+output blended), like a LiteLLM
# pricing table.
MODEL_COST_PER_1K = {
    CHEAP_MODEL: 0.002,
    STRONG_MODEL: 0.030,
}

COMPLEX_KEYWORDS = {
    "architecture", "design", "compare", "why", "explain", "trade-off",
    "tradeoff", "strategy", "optimi", "debug", "refactor", "analyze",
}


class ModelRouter:
    """
    Classifies query complexity cheaply, before spending a full model call,
    and also supports cascading (try cheap model, escalate on low confidence).
    """

    def classify(self, prompt: str) -> str:
        """Heuristic router: length + keyword signals (cheap, fast, no model call)."""
        text = prompt.lower()
        word_count = len(prompt.split())
        has_complex_kw = any(kw in text for kw in COMPLEX_KEYWORDS)
        if word_count > 40 or has_complex_kw:
            return STRONG_MODEL
        return CHEAP_MODEL

    def cascade_should_escalate(self, cheap_response_confidence: float, threshold: float = 0.6) -> bool:
        """Cascading: escalate to the strong model if the cheap model was unsure."""
        return cheap_response_confidence < threshold


# ---------------------------------------------------------------------------
# 5. MOCK PROVIDER (stand-in for LiteLLM's completion())
# ---------------------------------------------------------------------------

class MockLLMProvider:
    """
    Stand-in for `litellm.completion(model=..., messages=...)`. Simulates
    a provider response, a token count, and a confidence score (for
    cascading), all deterministic so the demo is reproducible.
    """

    def complete(self, model: str, prompt: str) -> tuple[str, int, float]:
        token_count = max(1, len(prompt.split()) + 20)  # + mock output tokens
        # Deterministic pseudo-confidence: cheap model is less confident on
        # longer / more complex-looking prompts, to make cascading kick in
        # realistically during the demo.
        base_confidence = 0.95 if model == STRONG_MODEL else 0.85
        length_penalty = min(0.4, len(prompt.split()) / 100)
        confidence = max(0.3, base_confidence - length_penalty)
        response = f"[{model} response] Answer to: {prompt[:60]}..."
        return response, token_count, confidence


# ---------------------------------------------------------------------------
# 6. COST TRACKING
# ---------------------------------------------------------------------------

@dataclass
class CostTracker:
    total_cost: float = 0.0
    cost_by_model: dict = field(default_factory=lambda: defaultdict(float))
    calls_by_model: dict = field(default_factory=lambda: defaultdict(int))
    cache_savings: float = 0.0
    requests_served_from_cache: int = 0

    def record_call(self, model: str, tokens: int):
        cost = (tokens / 1000) * MODEL_COST_PER_1K[model]
        self.total_cost += cost
        self.cost_by_model[model] += cost
        self.calls_by_model[model] += 1

    def record_cache_hit(self, would_be_model: str, would_be_tokens: int):
        saved = (would_be_tokens / 1000) * MODEL_COST_PER_1K[would_be_model]
        self.cache_savings += saved
        self.requests_served_from_cache += 1

    def report(self) -> str:
        lines = ["=== Cost Report ==="]
        lines.append(f"Total spend:            ${self.total_cost:.4f}")
        for model, cost in self.cost_by_model.items():
            lines.append(f"  {model:12s} -> ${cost:.4f} over {self.calls_by_model[model]} calls")
        lines.append(f"Estimated cache savings: ${self.cache_savings:.4f} "
                      f"({self.requests_served_from_cache} requests avoided an LLM call)")
        if self.total_cost + self.cache_savings > 0:
            pct_saved = self.cache_savings / (self.total_cost + self.cache_savings) * 100
            lines.append(f"Effective cost reduction: {pct_saved:.1f}%")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# THE GATEWAY -- ties everything together
# ---------------------------------------------------------------------------

class LLMGateway:
    def __init__(self, semantic_threshold: float = 0.5, verbose: bool = True):
        self.exact_cache = ExactMatchCache()
        self.semantic_cache = SemanticCache(similarity_threshold=semantic_threshold)
        self.compressor = PromptCompressor()
        self.router = ModelRouter()
        self.provider = MockLLMProvider()
        self.cost = CostTracker()
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def ask(self, prompt: str, temperature: float = 0.2) -> str:
        # Step 0: route BEFORE compressing/calling, so we know which model
        # this request "would" have cost, for cache-savings accounting.
        target_model = self.router.classify(prompt)

        # Step 1: exact-match cache check (cheapest, fastest lookup)
        cached = self.exact_cache.get(target_model, prompt, temperature)
        if cached is not None:
            self._log(f"[EXACT HIT]    -> {target_model}")
            self.cost.record_cache_hit(target_model, len(prompt.split()) + 20)
            return cached

        # Step 2: semantic cache check (falls through from exact-match miss)
        sem_hit = self.semantic_cache.get(target_model, prompt)
        if sem_hit is not None:
            response, score = sem_hit
            self._log(f"[SEMANTIC HIT] -> {target_model} (similarity={score:.2f})")
            self.cost.record_cache_hit(target_model, len(prompt.split()) + 20)
            # populate exact-match cache too, so a literal repeat is even faster next time
            self.exact_cache.set(target_model, prompt, temperature, response)
            return response

        # Step 3: prompt compression before the real call
        compressed_prompt, ratio = self.compressor.compress(prompt)
        if ratio > 1.0:
            self._log(f"[COMPRESSED]   {ratio:.2f}x ({len(prompt)} -> {len(compressed_prompt)} chars)")

        # Step 4: cascading -- try cheap model first regardless of router's
        # pick, ONLY when the router said "cheap" (complex prompts skip
        # straight to the strong model rather than wasting a cheap-model call).
        if target_model == CHEAP_MODEL:
            response, tokens, confidence = self.provider.complete(CHEAP_MODEL, compressed_prompt)
            self.cost.record_call(CHEAP_MODEL, tokens)
            if self.router.cascade_should_escalate(confidence):
                self._log(f"[CASCADE]      small-model confidence={confidence:.2f} -> escalating")
                response, tokens, confidence = self.provider.complete(STRONG_MODEL, compressed_prompt)
                self.cost.record_call(STRONG_MODEL, tokens)
                target_model = STRONG_MODEL
            else:
                self._log(f"[ROUTED]       -> {CHEAP_MODEL} (confidence={confidence:.2f}, no escalation)")
        else:
            response, tokens, confidence = self.provider.complete(STRONG_MODEL, compressed_prompt)
            self.cost.record_call(STRONG_MODEL, tokens)
            self._log(f"[ROUTED]       -> {STRONG_MODEL} (complex prompt, skipped cascade)")

        # Step 5: populate both caches for next time
        self.exact_cache.set(target_model, prompt, temperature, response)
        self.semantic_cache.set(target_model, prompt, response)

        return response

    def summary(self):
        print("\n" + self.cost.report())
        print(f"\nExact-match cache hit rate:  {self.exact_cache.hit_rate:.1%}")
        print(f"Semantic cache hit rate:     {self.semantic_cache.hit_rate:.1%}")


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gateway = LLMGateway(semantic_threshold=0.5)

    queries = [
        # exact-repeat pair -> should hit EXACT cache on the 2nd call
        "What is your refund policy?",
        "What is your refund policy?",

        # paraphrase of the above -> should hit SEMANTIC cache
        "What is the policy on refunds and returns?",

        # a genuinely different simple query
        "What are your support hours?",

        # a complex query -> should route straight to the strong model
        "Explain the trade-offs between microservice and monolith architecture "
        "for a payments platform handling 10k transactions per second.",

        # long, borderline-complex query -> good candidate to show cascading
        "I have a Python script that processes CSV files and it's running "
        "slowly on large files, can you help me debug why and suggest a fix "
        "please just walk me through the likely causes",

        # near-duplicate of the support-hours query -> semantic hit expected
        "When is your support team available?",
    ]

    print("=== Running demo queries through the gateway ===\n")
    for q in queries:
        print(f"> {q}")
        answer = gateway.ask(q)
        print(f"  {answer}\n")

    gateway.summary()