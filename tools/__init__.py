# Import and expose main tools from each module
from .spotify import *
from .spot import music_player_tools
from .notion_tools import notion_tools
from .gmail_tools import gmail_tools
from .calendar_tools import calendar_tools
from .tweet_test import twitter_tools
from .jigsaw import web_search_tool
from .screen import screen_tools
from .keyboard_tools import keyboard_tools

# Create a consolidated list of all tools
all_tools = [
    *music_player_tools,
    *notion_tools,
    *gmail_tools,
    *calendar_tools,
    *twitter_tools,
    web_search_tool,
    *screen_tools,
    *keyboard_tools
]