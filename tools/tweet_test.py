from typing import TypedDict, Sequence
import os
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import tweepy
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

client = tweepy.Client(
    consumer_key=os.getenv('consumer_key'), consumer_secret=os.getenv('consumer_secret'),
    access_token=os.getenv('access_token'), access_token_secret=os.getenv('access_token_secret')
)

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
)

# Define Twitter tools
def post_tweet(content: str) -> str:
    """Post a tweet to Twitter"""
    try:
        response = client.create_tweet(text=content)
        tweet_id = response.data['id']
        return f"Tweet posted successfully! URL: https://twitter.com/user/status/{tweet_id}"
    except Exception as e:
        return f"Error posting tweet: {str(e)}"

def draft_tweet(topic: str) -> str:
    """Draft a tweet about a specific topic"""
    prompt = f"""Draft a tweet about: {topic}
    Requirements:
    - Keep it under 280 characters
    - Make it engaging and natural
    - Don't use hashtags unless specifically requested
    - Be professional and friendly
    """
    response = llm.invoke(prompt)
    return response.content

class TweetContent(BaseModel):
    content: str = Field(description="The content of the tweet to post")

class TweetTopic(BaseModel):
    topic: str = Field(description="The topic to draft a tweet about")

# Create structured tools
twitter_tools = [
    StructuredTool(
        name="post_tweet",
        description="Post a tweet to Twitter",
        func=post_tweet,
        args_schema=TweetContent
    ),
    StructuredTool(
        name="draft_tweet",
        description="Draft a tweet about a topic",
        func=draft_tweet,
        args_schema=TweetTopic
    )
]