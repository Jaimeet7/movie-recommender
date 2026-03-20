import streamlit as st
import pandas as pd

from src.parse import parse_query, reference_movie
from src.search import apply_filters
from src.semantic_search import create_embeddings, similarity_search
from src.data_loader import load_data

import os
from dotenv import load_dotenv
load_dotenv()
datapath = os.getenv("DATA_PATH")

@st.cache_resource(show_spinner=True) #caching result without reloading dataset
def load_model_and_data(datapath: str):
    df = load_data(datapath)
    embeddings = create_embeddings(df)
    return df, embeddings


def main():
    st.set_page_config(
        page_title="Netflix Movie Recommender",
        page_icon="🎬",
        layout="wide",
    )

    st.markdown(
        "<h1 style='text-align: center;'>Netflix‑Style Movie Recommender</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; font-size: 1.05rem;'>"
        "Ask for movies or TV shows in natural language and get smart recommendations."
        "</p>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Number of recommendations", 3, 10, 5)
        st.markdown("---")
        st.caption("Type things like:")
        st.code("Suggest a Nolan movie\nIndian thriller series\nHorror movies with American cast")

    df, embedded_df = load_model_and_data(datapath)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat-like interface
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("What would you like to watch today?")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Finding the best matches for you..."):
            # Match CLI behavior for "like <movie>" queries
            ref_movie = reference_movie(prompt)
            if ref_movie:
                movie_row = df[df["title"].str.lower().str.contains(ref_movie.lower(), na=False)]
                if movie_row.empty:
                    reply = (
                        f"I couldn't find a title matching **{ref_movie}**."
                        "Try a different reference title'."
                    )
                    st.markdown(reply)
                    results = pd.DataFrame()
                else:
                    movie_index = movie_row.index[0]
                    ref_movie_embedding = embedded_df[movie_index]

                    # Exclude the reference movie itself from candidates
                    candidate_idx = df.index[df.index != movie_index]
                    df_no_ref = df.loc[candidate_idx]
                    embedded_no_ref = embedded_df[candidate_idx]

                    results = similarity_search(
                        ref_movie_embedding, df_no_ref, embedded_no_ref, top_k=top_k
                    )
            else:
                filters = parse_query(prompt)
                filtered_df = apply_filters(df, filters)

                if filtered_df.empty:
                    reply = "I couldn't find anything matching those filters. Try broadening your request."
                    st.markdown(reply)
                    results = pd.DataFrame()
                else:
                    filtered_indices = filtered_df.index
                    results = similarity_search(
                        prompt, filtered_df, embedded_df[filtered_indices], top_k=top_k
                    )

            if results.empty:
                # If we already rendered an explanation, keep it; otherwise show a generic one.
                reply = locals().get(
                    "reply", "No similar titles found. Please rephrase your query."
                )
                st.markdown(reply)
            else:
                # Render each recommendation as a card with emoji and left‑aligned title
                pieces = []
                for _, row in results.iterrows():
                    title = row["title"]
                    genre = row["listed_in"]
                    desc = row["description"]
                    score = float(row["scores"]) * 100

                    # For chat history markdown
                    pieces.append(
                        f"🎬 **{title}**  \n_{genre}_  \n{desc}  \n**Match score:** {score:.1f}%"
                    )

                    # On-screen layout: emoji on the left, text on the right
                    left, right = st.columns([1, 9])
                    with left:
                        st.markdown("🎬", unsafe_allow_html=True)
                    with right:
                        st.markdown(f"Title: **{title}**")
                        st.markdown(f"Genre: _{genre}_")
                        st.markdown(desc)
                        st.markdown(f"Score: **Match score:** {score:.2f}%")
                        st.markdown("---")

                reply = "\n\n---\n\n".join(pieces)

        st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()