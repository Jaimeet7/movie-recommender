from src.data_loader import load_data
from dotenv import load_dotenv
import os
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")

df = load_data(DATA_PATH)

def apply_filters(df,filters):
    res = df.copy() 
    
    if filters['type']:
        res = res.loc[res['type'].str.lower() == filters['type']]
    if filters['genre']:
        res = res.loc[res['listed_in'].str.contains(filters['genre'],case=False,na=False)]
    if filters['name']:
        res = res.loc[res['director'].str.contains(filters['name'],case=False,na=False) | res['cast'].str.contains(filters['name'],case=False,na=False)]
    if filters['country']:
        res=res.loc[res['country'].str.contains(filters['country'],case=False,na=False)]

    # Return the text column actually used for embeddings (`overview`)
    return res[['title','description','listed_in','overview']]