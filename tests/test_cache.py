import pytest

from gateway.cache import ResponseCache, exact_key
from gateway.config import settings
from gateway.embeddings import cosine, embed
from gateway.store import MemoryStore

PARAMS = {"temperature": 0.2, "max_tokens": 512}


def test_exact_key_is_stable_and_normalised():
    a = exact_key("m", "Hello   World", PARAMS)
    b = exact_key("m", "hello world", PARAMS)
    assert a == b, "whitespace and case must normalise to the same key"


def test_exact_key_separates_models_and_params():
    assert exact_key("m1", "p", PARAMS) != exact_key("m2", "p", PARAMS)
    assert exact_key("m", "p", PARAMS) != exact_key("m", "p", {**PARAMS, "max_tokens": 8})


def test_embeddings_are_normalised_and_deterministic():
    v1, v2 = embed("same text"), embed("same text")
    assert v1.tolist() == v2.tolist()
    assert abs(cosine(v1, v1) - 1.0) < 1e-5


def test_paraphrase_scores_above_unrelated():
    base = embed("What is the time complexity of binary search")
    para = embed("what is the time complexity of binary search?")
    other = embed("How does garbage collection work in the JVM")
    assert cosine(base, para) > cosine(base, other)
    assert cosine(base, para) >= settings.semantic_threshold
    assert cosine(base, other) < settings.semantic_threshold


@pytest.mark.asyncio
async def test_exact_hit_round_trip():
    cache = ResponseCache(MemoryStore())
    assert await cache.get("m", "prompt one", PARAMS) is None
    await cache.put("m", "prompt one", PARAMS, {"text": "x", "tokens": 10})
    hit = await cache.get("m", "prompt one", PARAMS)
    assert hit is not None and hit.tier == "exact" and hit.similarity == 1.0


@pytest.mark.asyncio
async def test_semantic_hit_on_paraphrase():
    cache = ResponseCache(MemoryStore())
    original = "Explain eventual consistency in distributed systems"
    await cache.put("m", original, PARAMS, {"text": "answer", "tokens": 25})
    hit = await cache.get("m", original + "?", PARAMS)
    assert hit is not None
    assert hit.tier == "semantic"
    assert hit.similarity >= settings.semantic_threshold


@pytest.mark.asyncio
async def test_unrelated_prompt_does_not_hit():
    cache = ResponseCache(MemoryStore())
    await cache.put("m", "Explain how a bloom filter works", PARAMS,
                    {"text": "a", "tokens": 5})
    assert await cache.get("m", "What is the difference between TCP and UDP", PARAMS) is None


@pytest.mark.asyncio
async def test_semantic_index_respects_cap():
    original = settings.semantic_index_cap
    settings.semantic_index_cap = 3
    try:
        cache = ResponseCache(MemoryStore())
        for i in range(10):
            await cache.put("m", f"distinct prompt number {i}", PARAMS,
                            {"text": str(i), "tokens": 1})
        assert cache.index_size <= 3
    finally:
        settings.semantic_index_cap = original
