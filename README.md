## Netflix Movie Recommender

Interactive movie and TV show recommender demonstrating semantic search, 
NLP-based query parsing with spaCy NER, and a RAG-inspired retrieval pipeline — 
built on the Netflix titles dataset and exposed via both a CLI and a Streamlit chatbot UI.

### Features

- **Natural‑language queries**: Ask for things like “Suggest a Nolan movie”, “Indian thriller series”, or “Horror movies with American cast”.
- **Rule‑based filters**: Simple NLP in `src/parse.py` extracts filters (type, genre, director, country, actor).
- **Semantic search**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to embed overviews and compute cosine similarity.
- **Multiple interfaces**:
  - CLI loop in `main.py`.
  - Chatbot‑style web UI in `streamlit_app.py`.

### Project structure

- `main.py` – Command‑line entrypoint using your parsing, filtering, and semantic search pipeline.
- `streamlit_app.py` – Streamlit chatbot UI for interactive recommendations.
- `data/netflix_cleaned.csv` – Cleaned Netflix titles dataset (contains `overview` text column).
- `src/parse.py` – Heuristic parser that converts free‑text queries into a `filters` dict.
- `src/search.py` – Applies filters to the dataframe (type / genre / director / country / actor).
- `src/semantic_search.py` – Creates embeddings and runs cosine‑similarity‑based retrieval.
- `src/data_loader.py` – Helper to load the CSV; used by `main.py` and the Streamlit app.

### Setup

1. **Create / activate virtual environment (optional but recommended)**:

```bash
python3 -m venv movie_env
source movie_env/bin/activate  # on macOS / Linux
# .\movie_env\Scripts\activate  # on Windows
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Ensure data file is present**:

Place `netflix_cleaned.csv` under:

```text
data/netflix_cleaned.csv
```

4. **Create datafrme path in .env**

```.env
DATA_PATH = "/Users/jaimeet/Documents/Movie Recommender/data/netflix_cleaned.csv"
```

and make sure it contains an `overview` column with the combined text for each title.

### Running the CLI recommender

From the project root:

```bash
python main.py
```

Then type queries like:

- `Suggest a nolan movie`
- `Indian thriller series`
- `Horror movies`

Type `exit` to quit the CLI loop.

### Running the Streamlit chatbot UI

From the project root:

```bash
streamlit run streamlit_app.py
```

Then open the URL printed in the terminal (by default `http://localhost:8501`) and chat with the app using natural‑language requests.

### Example Queries

- `Suggest movies with tom cruise`
- `Suggest a k drama`
- `Suggest movies like inception`

### How it works (high level)

1. **Data loading**: `load_data` reads `netflix_cleaned.csv` into a pandas dataframe.
2. **Embeddings**: `create_embeddings` encodes the `overview` text for each row with `SentenceTransformer("all-MiniLM-L6-v2")`.
3. **Query parsing**: `parse_query` extracts structured filters from the user query 
(type, genre, director, country, actor) using a combination of rule-based matching 
and spaCy NER (`en_core_web_lg`). One key insight during development: spaCy's NER 
relies on proper casing to recognize person entities, so queries are title-cased 
before NER runs and extracted names are lowercased afterward for consistent matching. 
Country extraction uses a nationality map (e.g. "korean" → "South Korea") as a 
fallback when NER returns a NORP entity.
4. **Filtering**: `apply_filters` narrows the dataframe to only matching rows.
5. **Semantic ranking**: `similarity_search` embeds the user query, computes cosine similarity against the precomputed embeddings, and returns the top‑k titles with similarity scores.

### Future Imporvements
1. FAISS — replace sklearn cosine similarity for production scale with millions of titles
2. LLM-based query parsing — replace rule-based spaCy parser with an LLM call that extracts structured filters, handles any phrasing, typos, and edge cases automatically
3. Spell correction — correct misspelled actor/director names before NER runs
4. Year/decade filtering — support queries like "90s thriller" or "movies from 2010"
5. Expanded nationality map — currently covers major countries, could be extended or replaced with a more robust geopolitical NLP model

### Notes

- The first run will download model weights from the Hugging Face Hub; subsequent runs will be faster.