import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from typing import Optional
import os

load_dotenv()

class filters(BaseModel):
    type:Optional[str] = Field(default=None,description="retrieve type like movie or series")
    genre:Optional[str] = Field(default=None,description="retrieve genre from query")
    name:Optional[str] = Field(default=None,description="retrieve any actor or director name in the query")
    country:Optional[str] = Field(default=None,description="retrieve country/origin/nationality of movie or series mentioned in query")

parser = PydanticOutputParser(pydantic_object=filters)


GOOGLE_API_KEY = os.getenv("API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0,google_api_key=GOOGLE_API_KEY)

prompts = PromptTemplate(
    template = "Extract movie filter for this query. \n{format_instructions}\nQuery:{query}",
    input_variables=["query"],
    partial_variables={"format_instructions":parser.get_format_instructions()}
)

chain = prompts | llm | parser

def parse_query_llm(query,chain):
    response = chain.invoke({'query':query})
    filters = {
        'type':response.type,
        'genre':response.genre,
        'name':response.name,
        'country':response.country
    }
    return filters

print(parse_query_llm("Suggest movie with tom cruise",chain))