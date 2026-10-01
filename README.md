# 🎯 Tech Stack Recommender

**DecodeLabs · Artificial Intelligence Track · Batch 2026 · Project 3: AI Recommendation Logic**

A web app that maps a user's skills to the best-matching tech career paths using **content-based filtering** with **TF-IDF** and **cosine similarity**.

## How it works (Input → Process → Output)

| Step | What happens |
|---|---|
| **1. Ingestion** | The user picks at least 3 skills. Inputs are cleaned and matched to the dataset vocabulary (e.g. `ML` → `Machine Learning`, `k8s` → `Kubernetes`). |
| **2. Vector mapping** | Every job role and the user profile become numeric vectors in one shared skill vocabulary. Multi-word skills such as "Machine Learning" stay as single tokens. |
| **3. TF-IDF weighting** | Rare, specific skills count for more than skills shared by many roles. |
| **4. Cosine similarity** | The angle between the user vector and each role vector gives a score between 0 and 1. |
| **5. Sort and filter** | Roles are ranked by score and the Top-N (default 3) are displayed. |

**Cold start handling:** if fewer than 3 recognised skills are given, the app shows trending roles instead of failing. Unrecognised inputs are reported rather than silently dropped.

**Explainability:** each result shows the skills you already have for the role and the highest-value skills to learn next.

## Features
- Skill picker plus free-text input with abbreviation support
- Top-N slider (1 to 5), match percentage and skill-gap guidance
- Score chart across all roles
- Built-in accuracy tab (see below)

## Dataset
`raw_skills.csv` contains **24 job roles** and **70 unique skills** across Data & AI, Software, Cloud & Infrastructure, Security and Design.
Columns: `role, category, popularity, description, skills` (skills separated by `;`).

> The dataset is **illustrative and was created for this project**. `popularity` is a simple 1 to 10 weight used only for the trending fallback and tie-breaking. It is not real market data. To use another dataset, keep the same columns.

## Accuracy check
For every role, 3 to 5 random skills **from that role** are given to the system (100 trials per role, seed 42), and we check whether the role is returned.

| Skills given | Top-3 hit rate | Top-1 hit rate |
|---|---|---|
| 3 | 99.6% | 81.5% |
| 4 | 100% | 90.7% |
| 5 | 100% | 96.6% |

This is a self-consistency test on an illustrative dataset, not a real-world benchmark. Top-1 is lower for closely related roles (for example NLP Engineer, MLOps Engineer, Site Reliability Engineer) because they share many skills.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy for free (Streamlit Community Cloud)
1. Push all files in this repository to GitHub.
2. Go to **share.streamlit.io** and sign in with GitHub.
3. Choose **Create app**, select this repository, set the main file to `app.py`, and click **Deploy**.

## Project structure
```
├── app.py             # Streamlit web interface
├── recommender.py     # TF-IDF + cosine similarity engine
├── raw_skills.csv     # Roles and skills dataset
├── requirements.txt
└── README.md
```

## Tech stack
Python · scikit-learn · pandas · NumPy · Streamlit
App link:https://teck-stack-recommender-project-3-bzzfwkh2bdd77sljxkeuto.streamlit.app/
## Author
**I. Rihana Sulthana**
AI Intern, DecodeLabs (Batch 2026)
