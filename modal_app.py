import modal

app = modal.App("course-recommendation")

# Cloud environment
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers",
        "numpy",
        "fastapi[standard]",
    )
    .add_local_python_source("catalogue", "recommend")
)


@app.cls(
    image=image,
    cpu=2,
    memory=2048,
    timeout=300,
)
class CourseRecommender:

    @modal.enter()
    def load_model(self):
        print("Loading all-MiniLM-L6-v2...")

        from sentence_transformers import SentenceTransformer
        import numpy as np

        from catalogue import COURSES

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.courses = COURSES

        catalogue_texts = [
            f"{c['title']}. {c['description']}"
            for c in self.courses
        ]

        self.catalogue_vectors = np.array(
            self.model.encode(catalogue_texts)
        )

        print(
            "Catalogue embeddings ready.",
            self.catalogue_vectors.shape,
        )

    @modal.fastapi_endpoint(method="GET")
    def recommend(
        self,
        gap_description: str,
        official_current_level: int,
        top_k: int = 3,
    ):
        from recommend import recommend_courses

        results = recommend_courses(
            gap_description=gap_description,
            official_current_level=official_current_level,
            model=self.model,
            catalogue_courses=self.courses,
            catalogue_vectors=self.catalogue_vectors,
            top_k=top_k,
        )

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
@app.local_entrypoint()
def main():
    print("Modal app loaded successfully!")