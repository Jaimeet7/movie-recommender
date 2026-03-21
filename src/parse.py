import spacy
import re

nlp = spacy.load("en_core_web_lg")

nationality_map = {
    "India": ["indian", "india","hindi"],
    "United States": ["american", "america", "usa", "us"],
    "United Kingdom": ["british", "uk", "england"],
    "South Korea":["korean","korea","south korea","k-drama","k drama"],
    "Canada":["canadian","canada"],
    "Japan":['japanese',"anime"]
}
genre_map = {
    "comedy": ["comedy","funny", "humor", "humorous"],
    "horror": ["horror","scary", "creepy", "terrifying"],
    "romance": ["romance","romantic", "love"],
    "thriller": ["thriller","suspense", "suspenseful"],
    "sci-fi": ["sci-fi","space", "science fiction", "sci fi"],
    "action": ["action","action packed"],
    "drama": ["drama","emotional", "sad"]
}

def check_name_in_query(query,df,filters):
    words = [word for word in query.lower().split() if len(word)>2]

    for n in range(len(words),0,-1):
        for i in range(len(words)-n+1):
            phrase = ' '.join(words[i:i+n])
            director_match = df[df['director'].str.contains(phrase,case=False,na=False)]
            if not director_match.empty:
                filters['name'] = phrase
                return filters
            cast_match = df[df['cast'].str.contains(phrase,case=False,na=False)]
            if not cast_match.empty:
                filters['name'] = phrase
                return filters
    return filters

def parse_query(query,df):

    filters = {
        "type" : None,
        "genre" : None,
        "name": None,
        "country": None,
    }

    query = query.lower()

    if "tv show" in query:
        filters['type'] = "tv show"
    if "movie" in query: 
        filters['type'] = "movie"

    genre = get_genre(query)
    if genre:
        filters['genre'] = genre

    for country,variants in nationality_map.items():
        if any(v in query for v in variants):
            filters['country'] = country
            break

    check_name_in_query(query,df,filters)

    ner_filter(query,filters)
    
    return filters


def ner_filter(query,filters):
    doc = nlp(query.title())
    
    for ent in doc.ents:
        if ent.label_ == "GPE":
            filters['country'] = ent.text
        elif ent.label_ == 'NORP':
            country = get_country(ent.text)
            if country:
                filters['country'] = country
    return filters

# print(ner_filter("Suggest a nolan movie",d))

def get_country(word):
    word = word.lower()
    for country,variants in nationality_map.items():
        if word in variants:
            return country

def reference_movie(query):
    query = query.lower()

    pattern = r"(?:like|similar to|similar)\s+(?:the\s+)?(?:movie\s+|film\s+)?(.+)"

    match = re.search(pattern,query)

    if match:
        movie = match.group(1).strip()
        return movie
    return None


def get_genre(query):
    for genre,word in genre_map.items():
        for w in word:
            if w in query:
                return genre