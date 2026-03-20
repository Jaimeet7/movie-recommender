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
    "drama": ["drama","emotional", "sad","k"]
}

def parse_query(query):

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

    ner_filter(query,filters)
    
    return filters


def ner_filter(query,filters):
    doc = nlp(query.title())
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            filters['name'] = ent.text.lower()
        elif ent.label_ == "GPE":
            filters['country'] = ent.text
        elif ent.label_ == 'NORP':
            country = get_country(ent.text)
            if country:
                filters['country'] = get_country(country)
    return filters


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
        print(match.group().strip())
        return movie
    return None


def get_genre(query):
    for genre,word in genre_map.items():
        for w in word:
            if w in query:
                return genre
