from src.parse import parse_query,reference_movie
from src.search import apply_filters
from src.semantic_search import create_embeddings,similarity_search
from src.data_loader import load_data
from src.llm_parser import parse_query_llm,chain
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
datapath = os.getenv("DATA_PATH")

df = load_data(datapath)
embedded_df = create_embeddings(df)
generic_queries = [
    "what should i watch",
    "anything good",
    "any movie",
    "something to watch"
]

def keyword_search(df,query,embedded_df):
    choice = input("Since this is a project to display both LLM parser plus a Manual Parser please choose which would u prefer\n1.LLM parser\n2.Manual Parse\n")
    match choice:
        case '1': 
            filters = parse_query_llm(query,chain)
        case '2':
            filters = parse_query(query,df)
    filled = [v for v in filters.values() if v is not None]
    if len(filled)<2:
        print("Please provide a more descriptive query")
    else:
        filtered_df = apply_filters(df,filters)
        if filtered_df.empty:
            return None
        filtered_indices = filtered_df.index
        result = similarity_search(query,filtered_df,embedded_df[filtered_indices])
        return result

while True:
    query = input("What would you like to watch today?? \n")

    if any(q in query.lower() for q in generic_queries):
        result = df.sample(5)
        print("\n Here are some suggestions:\n")
        for _,row in result.iterrows():
                print(f"\nTitle: {row['title']}")
                print(f"\nGenre: {row['listed_in']}")
                print(f"\nDescription: {row['description']}")
        continue
    if query.lower() == "exit":
        break
    ref_movie = reference_movie(query)
    if ref_movie:
        movie_row = df[df['title'].str.contains(ref_movie.lower(),case=False)]
        if not movie_row.empty:
            movie_index = movie_row.index[0]
            ref_movie_embedding = embedded_df[movie_index]
            candidate_idx = df.index[df.index != movie_index]
            df_no_ref = df.loc[candidate_idx]
            embedded_no_ref = embedded_df[candidate_idx]

            result = similarity_search(ref_movie_embedding, df_no_ref, embedded_no_ref)
        else:
            print("Could not find that reference movie. Falling back to normal search.")
            result = keyword_search(df,query,embedded_df)
    else:
        result = keyword_search(df,query,embedded_df)
    if result is None or result.empty:
        print("No results found")
    else:
        for _,row in result.iterrows():
            print(f"\nTitle: {row['title']}")
            print(f"\nGenre: {row['listed_in']}")
            print(f"\nDescription: {row['description']}")
            print(f"\nScore:{row['scores']*100:.2f}%")