"""Content-based Tech Stack Recommender (TF-IDF + cosine similarity)."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Maps common spellings / abbreviations (lower-case) to the canonical skill name used in the dataset.
SYNONYMS = {
    "ml": "Machine Learning", "machine-learning": "Machine Learning",
    "dl": "Deep Learning",
    "js": "JavaScript", "javascript": "JavaScript", "ts": "TypeScript",
    "reactjs": "React", "react.js": "React", "node": "Node.js", "nodejs": "Node.js",
    "k8s": "Kubernetes", "tf": "TensorFlow", "sklearn": "Scikit-learn", "scikit learn": "Scikit-learn",
    "cloud": "Cloud Computing", "cloud computing": "Cloud Computing",
    "web design": "UI/UX Design", "ui": "UI/UX Design", "ux": "UI/UX Design", "ui/ux": "UI/UX Design",
    "frontend development": "HTML", "front end": "HTML",
    "data viz": "Data Visualization", "visualization": "Data Visualization",
    "powerbi": "Power BI", "power-bi": "Power BI", "postgres": "PostgreSQL", "mongo": "MongoDB",
    "ci cd": "CI/CD", "cicd": "CI/CD", "bash": "Bash Scripting", "shell scripting": "Bash Scripting",
    "api": "REST APIs", "apis": "REST APIs", "rest": "REST APIs", "rest api": "REST APIs",
    "dsa": "Data Structures", "data structure": "Data Structures", "algorithm": "Algorithms",
    "pen testing": "Penetration Testing", "pentesting": "Penetration Testing",
    "security": "Cybersecurity", "cyber security": "Cybersecurity", "network": "Networking",
    "large language models": "LLMs", "llm": "LLMs", "gen ai": "LLMs", "prompting": "Prompt Engineering",
    "gcp": "GCP", "google cloud": "GCP", "amazon web services": "AWS", "ms excel": "Excel",
    "stats": "Statistics", "analytics": "Data Analysis", "data analytics": "Data Analysis",
    "numpy": "NumPy", "pandas": "Pandas", "sql": "SQL", "nlp": "NLP", "mlops": "MLOps",
}


def _identity(tokens):
    """Analyzer for TfidfVectorizer: documents are already lists of skill tokens."""
    return tokens


def _split(skill_string: str) -> list[str]:
    return [s.strip() for s in str(skill_string).split(";") if s.strip()]


class TechStackRecommender:
    MIN_SKILLS = 3

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.role_skills = [_split(s) for s in self.df["skills"]]
        self.vocabulary = sorted({s for skills in self.role_skills for s in skills})
        self._canonical = {s.lower(): s for s in self.vocabulary}

        # One shared vocabulary for roles and users. Multi-word skills stay as single tokens.
        self.vectorizer = TfidfVectorizer(analyzer=_identity)
        self.role_matrix = self.vectorizer.fit_transform(self.role_skills)
        self.idf = dict(zip(self.vectorizer.get_feature_names_out(), self.vectorizer.idf_))

    # ---------- Step 1: ingestion ----------
    def normalize(self, raw: str) -> str | None:
        key = re.sub(r"\s+", " ", raw.strip().lower())
        if not key:
            return None
        if key in SYNONYMS:
            return SYNONYMS[key]
        return self._canonical.get(key)

    def parse(self, items: list[str]) -> tuple[list[str], list[str]]:
        """Return (recognised skills without duplicates, unrecognised inputs)."""
        known, unknown = [], []
        for raw in items:
            if not raw.strip():
                continue
            canon = self.normalize(raw)
            if canon and canon in self.vocabulary:
                if canon not in known:
                    known.append(canon)
            else:
                unknown.append(raw.strip())
        return known, unknown

    # ---------- Steps 2-4: scoring, sorting, filtering ----------
    def recommend(self, skills: list[str], top_n: int = 3) -> pd.DataFrame:
        user_vec = self.vectorizer.transform([skills])
        scores = cosine_similarity(user_vec, self.role_matrix).ravel()
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], -self.df.loc[i, "popularity"]))
        user_set = set(skills)
        rows = []
        for i in order[:top_n]:
            role_set = self.role_skills[i]
            matched = [s for s in role_set if s in user_set]
            missing = sorted((s for s in role_set if s not in user_set), key=lambda s: -self.idf[s])
            rows.append({
                "role": self.df.loc[i, "role"], "category": self.df.loc[i, "category"],
                "score": float(scores[i]), "matched": matched, "missing": missing,
                "description": self.df.loc[i, "description"],
            })
        return pd.DataFrame(rows)

    def all_scores(self, skills: list[str]) -> pd.Series:
        scores = cosine_similarity(self.vectorizer.transform([skills]), self.role_matrix).ravel()
        return pd.Series(scores, index=self.df["role"]).sort_values(ascending=False)

    # ---------- Cold-start fallback ----------
    def trending(self, top_n: int = 3) -> pd.DataFrame:
        return self.df.sort_values("popularity", ascending=False).head(top_n)[["role", "category", "description"]]

    # ---------- Evaluation ----------
    def evaluate(self, n_given: int = 3, top_n: int = 3, trials: int = 100, seed: int = 42) -> pd.DataFrame:
        """For each role, give the system `n_given` random skills of that role and check
        whether the role appears in the Top-N results."""
        rng = np.random.default_rng(seed)
        out = []
        for i, role in enumerate(self.df["role"]):
            pool = self.role_skills[i]
            hit_n = hit_1 = 0
            for _ in range(trials):
                pick = list(rng.choice(pool, size=min(n_given, len(pool)), replace=False))
                top = self.recommend(pick, top_n)["role"].tolist()
                hit_n += role in top
                hit_1 += top[0] == role
            out.append({"role": role, f"top{top_n}_hit_rate": hit_n / trials, "top1_hit_rate": hit_1 / trials})
        return pd.DataFrame(out)
