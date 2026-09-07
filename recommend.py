import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors, in [-1, 1].

    Returns 0.0 if either vector has zero magnitude (degenerate input)
    instead of raising a division-by-zero error.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def rank_by_similarity(
    query_vector: np.ndarray,
    courses: list[dict],
    course_vectors: np.ndarray,
) -> list[dict]:
    """Score every course against the query vector, best first.

    `courses[i]` must correspond to `course_vectors[i]`.
    Returns copies of the course dicts with a `similarity` field added.
    Does not mutate the input list.
    """
    scored = []
    for course, vec in zip(courses, course_vectors):
        score = cosine_similarity(query_vector, vec)
        scored.append({**course, "similarity": score})
    scored.sort(key=lambda c: c["similarity"], reverse=True)
    return scored


def filter_by_level(
    scored_courses: list[dict],
    official_current_level: int,
    max_level_ahead: int = 1,
) -> list[dict]:
    """Drop courses whose `level` is too far above the official's level.

    A course is kept if:
        course["level"] <= official_current_level + max_level_ahead

    This also drops nothing on the *low* side — an official at level 1
    can still see level-1 or level-2 courses, but not level-4 ones (with
    the default max_level_ahead=1). Courses below the official's own
    level are left in deliberately: someone who's a bit rusty on the
    basics (or whose profile data is incomplete) shouldn't be silently
    blocked from catching up.

    Fallback: if the filter would remove every single course, it's
    almost certainly because the level data is missing/inconsistent for
    this particular gap, not because "nothing is suitable" — in that
    case we return the unfiltered list rather than an empty one, since
    an unfiltered-but-ranked list is more useful than nothing.
    """
    ceiling = official_current_level + max_level_ahead
    filtered = [c for c in scored_courses if c.get("level", 0) <= ceiling]
    if not filtered:
        return scored_courses
    return filtered


def mmr_rerank(
    query_vector: np.ndarray,
    candidates: list[dict],
    candidate_vectors: np.ndarray,
    top_k: int = 3,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Maximal Marginal Relevance re-ranking.

    At each step, picks the candidate that maximizes:

        lambda * sim(query, candidate)
        - (1 - lambda) * max( sim(candidate, already_picked) )

    lambda_param close to 1.0  -> mostly relevance, diversity barely matters
    lambda_param close to 0.0  -> mostly diversity, relevance barely matters
    0.7 is a reasonable default: favor relevance, but still break ties
    between near-duplicate courses.

    `candidates[i]` must correspond to `candidate_vectors[i]`, and each
    candidate dict is expected to already have a `similarity` field
    (from rank_by_similarity) giving its relevance to the query.
    """
    if len(candidates) <= top_k:
        return candidates

    query_vector = np.asarray(query_vector, dtype=np.float32)
    candidate_vectors = np.asarray(candidate_vectors, dtype=np.float32)

    remaining = list(range(len(candidates)))
    selected: list[int] = []

    while remaining and len(selected) < top_k:
        best_idx = None
        best_score = -np.inf

        for idx in remaining:
            relevance = candidates[idx]["similarity"]

            if selected:
                redundancy = max(
                    cosine_similarity(candidate_vectors[idx], candidate_vectors[s])
                    for s in selected
                )
            else:
                redundancy = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def recommend_courses(
    gap_description: str,
    official_current_level: int,
    model,
    catalogue_courses: list[dict],
    catalogue_vectors: np.ndarray,
    top_k: int = 3,
    max_level_ahead: int = 1,
    mmr_lambda: float = 0.7,
    mmr_pool_size: int = 8,
) -> list[dict]:
    """End-to-end recommendation for a single skill gap.

    Steps:
        1. Embed the gap description with the (already-loaded) model.
        2. Rank every course in the catalogue by cosine similarity.
        3. Filter out courses too far above the official's level.
        4. Take a small pool of the top matches (mmr_pool_size) and
           re-rank that pool with MMR so the final top_k isn't three
           near-duplicates.

    Returns a list of up to `top_k` course dicts, each with a
    `similarity` field, best/most-diverse first.
    """
    query_vector = model.encode(gap_description)

    ranked = rank_by_similarity(query_vector, catalogue_courses, catalogue_vectors)
    filtered = filter_by_level(ranked, official_current_level, max_level_ahead)

    # Re-embed the surviving pool's vectors in the same order as `filtered`
    # so MMR compares apples to apples. We rebuild a lookup by id rather
    # than re-encoding text (cheap, avoids a second model call).
    vector_by_id = {c["id"]: v for c, v in zip(catalogue_courses, catalogue_vectors)}

    pool = filtered[:mmr_pool_size]
    pool_vectors = np.array([vector_by_id[c["id"]] for c in pool])

    reranked = mmr_rerank(
        query_vector,
        pool,
        pool_vectors,
        top_k=top_k,
        lambda_param=mmr_lambda,
    )
    return reranked