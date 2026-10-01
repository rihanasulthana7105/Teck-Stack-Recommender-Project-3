"""Tech Stack Recommender: DecodeLabs AI Project 3 (Streamlit web app)."""
import os
import streamlit as st
import pandas as pd
from recommender import TechStackRecommender

APP_NAME = "CareerCompass"
ICON = "icon.png" if os.path.exists("icon.png") else "🧭"   # falls back to an emoji if the file is missing

st.set_page_config(page_title=APP_NAME, page_icon=ICON, layout="centered")


@st.cache_resource
def load_engine():
    return TechStackRecommender("raw_skills.csv")


@st.cache_data
def run_evaluation(n_given):
    return load_engine().evaluate(n_given=n_given, top_n=3, trials=100, seed=42)


engine = load_engine()

st.title("🧭 " + APP_NAME)
st.caption("Your AI-powered Tech Stack Recommender. Tell us your skills and get the career paths that match them best. "
           "Content-based filtering with TF-IDF and cosine similarity.")

tab_rec, tab_scores, tab_how, tab_eval = st.tabs(["Recommend", "All roles", "How it works", "Accuracy"])

with tab_rec:
    st.subheader("1. Your skills")
    chosen = st.multiselect(
        f"Pick at least {engine.MIN_SKILLS} skills",
        options=engine.vocabulary,
        default=["Python", "Cloud Computing", "Automation"],
        help="Start typing to search the list.",
    )
    extra = st.text_input("Other skills (comma separated)", placeholder="e.g. ML, k8s, JS",
                          help="Common abbreviations such as ML, JS and k8s are understood.")
    top_n = st.slider("Number of recommendations", 1, 5, 3)

    extra_items = [x for x in extra.split(",") if x.strip()]
    skills, unknown = engine.parse(list(chosen) + extra_items)
    if unknown:
        st.info("Not in our dataset, so they were ignored: " + ", ".join(unknown))

    if st.button("Get recommendations", type="primary"):
        if len(skills) < engine.MIN_SKILLS:
            # Cold-start bypass: not enough data to personalise, so show trending roles instead.
            st.warning(f"Please give at least {engine.MIN_SKILLS} recognised skills for a personalised match. "
                       "Meanwhile, here are the most in-demand roles:")
            for _, row in engine.trending(top_n).iterrows():
                st.markdown(f"**{row['role']}** · {row['category']}  \n{row['description']}")
        else:
            st.subheader("2. Your best matches")
            results = engine.recommend(skills, top_n)
            if results["score"].max() == 0:
                st.error("None of your skills overlap with any role. Try different skills.")
            for rank, row in enumerate(results.itertuples(), start=1):
                with st.container(border=True):
                    st.markdown(f"### {rank}. {row.role}")
                    st.caption(row.category)
                    st.progress(min(row.score, 1.0), text=f"Match: {row.score:.0%}")
                    st.write(row.description)
                    st.markdown("✅ **You already have:** " + (", ".join(row.matched) or "none yet"))
                    st.markdown("📚 **Learn next:** " + ", ".join(row.missing[:5]))

with tab_scores:
    st.subheader("Match score for every role")
    st.caption("Uses the skills selected on the Recommend tab.")
    chosen_now, _ = engine.parse(list(chosen) + [x for x in extra.split(",") if x.strip()])
    if chosen_now:
        st.bar_chart(engine.all_scores(chosen_now).sort_values())
    else:
        st.write("Select some skills first.")

with tab_how:
    st.subheader("Pipeline (Input → Process → Output)")
    st.markdown("""
1. **Ingestion:** your skills are cleaned and matched to the dataset vocabulary (abbreviations like *ML* become *Machine Learning*). At least 3 skills are required.
2. **Vector mapping:** each job role and your profile become numeric vectors in one shared skill vocabulary.
3. **TF-IDF weighting:** rare, specific skills (e.g. *Kubernetes*) count for more than skills that appear in many roles (e.g. *Python*).
4. **Cosine similarity:** the angle between your vector and each role's vector gives a score from 0 to 1.
5. **Sort and filter:** roles are ranked by score and only the Top-N are shown.

**Cold start:** if there is not enough input, the app falls back to trending roles instead of failing.
""")
    st.markdown("**Dataset:** " + f"{len(engine.df)} roles, {len(engine.vocabulary)} skills.")
    st.dataframe(engine.df[["role", "category", "skills"]], hide_index=True)

with tab_eval:
    st.subheader("How accurate is it?")
    st.write("For every role, we give the system a few random skills *from that role* and check whether the role "
             "comes back in the Top 3 (100 trials per role).")
    n_given = st.select_slider("Skills given", options=[3, 4, 5], value=3)
    ev = run_evaluation(n_given)
    c1, c2 = st.columns(2)
    c1.metric("Top-3 hit rate", f"{ev['top3_hit_rate'].mean():.1%}")
    c2.metric("Top-1 hit rate", f"{ev['top1_hit_rate'].mean():.1%}")
    st.dataframe(ev.sort_values("top1_hit_rate"), hide_index=True)
    st.caption("This is a self-consistency check on an illustrative dataset, not a real-world benchmark.")
