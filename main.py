import numpy as np
from sentence_transformers import SentenceTransformer

from catalogue import COURSES
from recommend import recommend_courses

# ---------------------------------------------------------------------------
# Startup: runs once, when this module is first imported.
# ---------------------------------------------------------------------------

print("[Pipeline A] Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print(f"[Pipeline A] Embedding {len(COURSES)} courses from the catalogue...")
_catalogue_texts = [f"{c['title']}. {c['description']}" for c in COURSES]
CATALOGUE_VECTORS = np.array(model.encode(_catalogue_texts))
print("[Pipeline A] Catalogue embeddings ready. Vector dim:", CATALOGUE_VECTORS.shape[1])


def get_recommendations(
    gap_description: str,
    official_current_level: int,
    top_k: int = 3,
) -> list[dict]:
    """Per-request entry point. Cheap: only encodes the one gap string,
    everything else (model, catalogue vectors) is already in memory."""
    results = recommend_courses(
        gap_description=gap_description,
        official_current_level=official_current_level,
        model=model,
        catalogue_courses=COURSES,
        catalogue_vectors=CATALOGUE_VECTORS,
        top_k=top_k,
    )
    # Trim to the fields an API response actually needs.
    return [
        {
            "id": r["id"],
            "title": r["title"],
            "level": r["level"],
            "description": r["description"],
            "similarity": round(r["similarity"], 4),
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# Manual test harness — run `python main.py` to see this.
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def show(label, results):
        print(f"\n=== {label} ===")
        if not results:
            print("  (empty)")
        for r in results:
            print(f"  [{r['similarity']:.3f}] L{r['level']} {r['id']} {r['title']}")

    # --- Test 1: a clean, well-matched gap -----------------------------
    show(
        "Survey design gap, currently level 1",
        get_recommendations(
            "Survey Design gap, needs level 3, currently level 1",
            official_current_level=1,
        ),
    )

    # --- Test 2: a gap with basically nothing in the catalogue ----------
    # Deliberately unrelated to anything in the catalogue: espresso machines.
    show(
        "Nonsense/no-match gap",
        get_recommendations(
            "Needs to learn how to repair vintage espresso machines",
            official_current_level=1,
        ),
    )

    # --- Test 3: same gap, level dialed very low vs very high -----------
    show(
        "ML gap, official_current_level=1 (junior)",
        get_recommendations(
            "Officer needs machine learning and model evaluation skills",
            official_current_level=1,
        ),
    )
    show(
        "ML gap, official_current_level=4 (senior)",
        get_recommendations(
            "Officer needs machine learning and model evaluation skills",
            official_current_level=4,
        ),
    )

    # --- Test 4: two very similar gap descriptions -----------------------
    show(
        "Gap A: needs help with survey questionnaires",
        get_recommendations(
            "Officer struggles to write clear, unbiased survey questions",
            official_current_level=1,
        ),
    )
    show(
        "Gap B: questionnaire wording issues (near-duplicate of A)",
        get_recommendations(
            "Officer's questionnaire items are ambiguous and leading",
            official_current_level=1,
        ),
    )