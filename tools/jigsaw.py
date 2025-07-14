from jigsawstack import JigsawStack
from typing import Optional
from pydantic import BaseModel, Field
import os
from langchain.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
)

class WebSearchInput(BaseModel):
    """Input for web search."""
    query: str = Field(description="The search query to look up current information")

def web_search(query: str) -> str:
    """Search the web for current information."""
    jigsawstack = JigsawStack()  # API key will be read from environment
    result = jigsawstack.web.search({
        "query": query
    })
    prompt = f"""extract relevant information for this query: {query} from this {result}
    Requirements:
    keep it simple and in a conversation flow 
    """
    response = llm.invoke(prompt) 
    return response.content

# Define the tool
web_search_tool = StructuredTool(
    name="web_search",
    description="Search the web for current information like weather, news, or real-time data",
    func=web_search,
    args_schema=WebSearchInput
)
