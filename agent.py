from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from langgraph.prebuilt import create_react_agent

from typing import Dict, TypedDict, Annotated, Sequence, List, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import json
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

from enum import Enum
from typing import Optional
from langchain.tools import StructuredTool
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import threading
import asyncio
import keyboard
import speech_recognition as sr
import tempfile
from playsound import playsound
from time import sleep
import wave
import numpy as np
# Kokoro TTS imports
from kokoro import KPipeline
import soundfile as sf
import torch
import logging
import warnings
# Import all tools
#from tools import all_tools
# Or import specific tool groups
from tools import music_player_tools, notion_tools, gmail_tools, calendar_tools, twitter_tools, web_search_tool, screen_tools, keyboard_tools

# Import memory components
from memory_manager import MemorySystem, BackgroundMemoryProcessor

# Import procrastination components
from anti_procrastination import procrastination_check_tool, check_procrastination_notifications
# Import our new tools for screen analysis and keyboard control


# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress specific loggers
logging.getLogger('phonemizer').setLevel(logging.ERROR)
logging.getLogger('kokoro').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('numpy').setLevel(logging.ERROR)

import logging
from groq import Groq
import uuid
import numpy as np
from typing import Dict, List, Tuple, Any
import time
import queue

# Initialize Groq client for Whisper STT
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize text-to-speech engine with error handling
try:
    tts_pipeline = KPipeline(
        lang_code='a',  # 'a' for American English
        repo_id="hexgrad/Kokoro-82M",  # Explicitly specify repo
        device="cpu",  # Explicitly set device
    )
    print("Kokoro TTS initialized successfully")
    
    async def _speak_async(text):
        """Async function to generate and play speech with improved error handling"""
        temp_files_to_cleanup = []
        
        try:
            # Clean text for TTS
            text = text.strip()
            if not text:
                return
                
            # Using Kokoro TTS with optimized parameters
            generator = tts_pipeline(
                text, 
                voice='af_heart', 
                speed=1.2,  # Process one batch at a time to reduce memory usage
            )
            
            # Process and play each audio segment
            for i, (_, _, audio) in enumerate(generator):
                # Use tempfile module to safely create temporary files
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_path = temp_file.name
                    print(f"Creating temporary audio file: {temp_path}")
                
                try:
                    # Save audio data to the temporary file
                    sf.write(temp_path, audio, 24000)
                    print(f"Audio saved to temporary file, attempting playback...")
                    
                    # Check if file exists before playing
                    if not os.path.exists(temp_path):
                        raise FileNotFoundError(f"Temp file not found: {temp_path}")
                        
                    # Get absolute path to be safe
                    abs_path = os.path.abspath(temp_path)
                    
                    # Add to cleanup list instead of immediately removing
                    temp_files_to_cleanup.append(abs_path)
                    
                    # Play the audio with error handling
                    try:
                        playsound(abs_path)
                        print(f"Audio playback complete")
                    except Exception as play_error:
                        print(f"Playsound error: {str(play_error)}")
                        # Try alternative playback method if available
                        if os.name == 'nt':  # Windows
                            try:
                                import winsound
                                winsound.PlaySound(abs_path, winsound.SND_FILENAME)
                                print("Used winsound as fallback")
                            except Exception as win_error:
                                print(f"Windows playback fallback failed: {str(win_error)}")
                    
                except Exception as e:
                    print(f"Error during audio processing/playback: {str(e)}")
                    
        except KeyboardInterrupt:
            print("Speech interrupted")
        except Exception as e:
            print(f"TTS Error: {e}")
            print(f"Assistant (TTS Failed): {text}")
        
        finally:
            # Schedule cleanup for later to avoid Windows file locking issues
            if temp_files_to_cleanup:
                # For Windows: Create a deferred cleanup function
                if os.name == 'nt':
                    def cleanup_temp_files():
                        # Wait a bit for files to be released
                        time.sleep(0.5)
                        for file_path in temp_files_to_cleanup:
                            try:
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                                    print(f"Removed temp file: {file_path}")
                            except Exception as e:
                                # Just log errors but don't raise them
                                print(f"Note: Could not remove temp file {file_path} - will be cleaned up later: {e}")
                    
                    # Run cleanup in a separate thread to avoid blocking
                    threading.Thread(target=cleanup_temp_files, daemon=True).start()
                else:
                    # For non-Windows platforms, try immediate cleanup
                    for file_path in temp_files_to_cleanup:
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                        except Exception as e:
                            print(f"Note: Could not remove temp file {file_path}: {e}")
    
    def speak_response(text):
        """Synchronous wrapper for the async speech function"""
        try:
            asyncio.run(_speak_async(text))
        except Exception as e:
            print(f"TTS Error: {e}")
            print(f"Assistant (TTS Failed): {text}")
            
except Exception as e:
    print(f"Error initializing Kokoro TTS: {e}")
    # Fallback to simple print if TTS fails to initialize
    def _speak_async(text):
        print(f"TTS (Fallback): {text}")
    
    def speak_response(text):
        print(f"TTS (Fallback): {text}")

# Create a function to extract insights from conversation
def extract_memory_insights(messages: List[BaseMessage], memory: MemorySystem):
    """
    Extract insights from conversation to store in memory
    
    Parameters:
        messages: List of conversation messages
        memory: MemorySystem instance
    """
    if not messages or len(messages) < 2:
        return
    
    # Get the last user message and assistant response
    user_messages = [m for m in messages[-5:] if isinstance(m, HumanMessage)]
    assistant_messages = [m for m in messages[-5:] if isinstance(m, AIMessage)]
    
    if not user_messages or not assistant_messages:
        return
    
    last_user_msg = user_messages[-1]
    last_assistant_msg = assistant_messages[-1]
    
    # Store the interaction in episodic memory
    memory.store_episodic_memory({
        "content": f"User: {last_user_msg.content} → Assistant: {last_assistant_msg.content[:100]}...",
        "timestamp": datetime.now().isoformat(),
        "context": memory.working_memory.get("active_context", {})
    })
    
    # Extract potential semantic facts (very simple heuristic)
    user_text = last_user_msg.content.lower()
    
    # Check for preference indicators
    if "i like" in user_text or "i prefer" in user_text or "i want" in user_text or "i need" in user_text:
        memory.store_semantic_memory({
            "content": f"User preference: {last_user_msg.content}",
            "category": "preference"
        })
    
    # Extract tool usage patterns - with better error handling
    try:
        # Check if the message has tool_calls
        if hasattr(last_assistant_msg, "tool_calls") and last_assistant_msg.tool_calls:
            # Extract tool usage from direct tool_calls attribute
            for tool_call in last_assistant_msg.tool_calls:
                try:
                    # Try to get tool name - might be in different locations based on format
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("name", "unknown_tool")
                    else:
                        # Might be an object with attributes
                        tool_name = getattr(tool_call, "name", "unknown_tool")
                    
                    memory.update_procedural_memory(
                        "tools", 
                        tool_name, 
                        {
                            "prompt": last_user_msg.content,
                            "usage": "tool_call" # Just store a marker instead of the whole tool call
                        }
                    )
                except Exception as e:
                    print(f"Error processing individual tool call: {e}")
                    continue
        
        # Also check additional_kwargs for tool_calls (alternative format)
        elif hasattr(last_assistant_msg, "additional_kwargs") and "tool_calls" in last_assistant_msg.additional_kwargs:
            additional_tool_calls = last_assistant_msg.additional_kwargs["tool_calls"]
            for tool_call in additional_tool_calls:
                try:
                    tool_name = tool_call.get("function", {}).get("name", "unknown_tool")
                    memory.update_procedural_memory(
                        "tools", 
                        tool_name, 
                        {
                            "prompt": last_user_msg.content,
                            "usage": "tool_call"
                        }
                    )
                except Exception as e:
                    print(f"Error processing additional_kwargs tool call: {e}")
                    continue
    except Exception as e:
        print(f"Error extracting tool patterns: {e}")
    
    # Save memories periodically (every 10 interactions)
    if len(messages) % 10 == 0:
        memory.save_memories()

# Memory tools for the agent
def create_memory_tools(memory_system: MemorySystem):
    """Create tools for the agent to interact with memory"""
    
    # Tool to search memory
    def search_memory(
        memory_type: str = "episodic", 
        query: str = "", 
        limit: int = 5
    ) -> str:
        """
        Search the assistant's memory for relevant information
        
        Args:
            memory_type: Type of memory to search (episodic, semantic, procedural)
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            String with the search results
        """
        if memory_type == "episodic":
            results = memory_system.search_episodic_memory(query, k=limit)
            return "\n".join([f"Memory {i+1}: {doc.page_content}" for i, doc in enumerate(results)])
        
        elif memory_type == "semantic":
            results = memory_system.search_semantic_memory(query, k=limit)
            return "\n".join([f"Fact {i+1}: {doc.page_content}" for i, doc in enumerate(results)])
            
        elif memory_type == "procedural":
            pattern_types = ["tools", "workflows", "preferences", "response_patterns"]
            results = []
            for pattern_type in pattern_types:
                patterns = memory_system.get_procedural_pattern(pattern_type, query)
                if patterns:
                    pattern_results = [f"{pattern_type.capitalize()} pattern: {k}" for k in list(patterns.keys())[:limit]]
                    results.extend(pattern_results)
            return "\n".join(results)
            
        elif memory_type == "all":
            episodic = memory_system.search_episodic_memory(query, k=limit)
            semantic = memory_system.search_semantic_memory(query, k=limit)
            
            results = []
            if episodic:
                results.append("Past interactions:")
                results.extend([f"- {doc.page_content}" for doc in episodic])
            
            if semantic:
                results.append("\nFacts and preferences:")
                results.extend([f"- {doc.page_content}" for doc in semantic])
                
            # Add procedural patterns
            pattern_types = ["tools", "workflows"]
            procedural_results = []
            for pattern_type in pattern_types:
                patterns = memory_system.get_procedural_pattern(pattern_type, query)
                if patterns:
                    procedural_results.append(f"\n{pattern_type.capitalize()}:")
                    procedural_results.extend([f"- {k}" for k in list(patterns.keys())[:limit]])
            
            results.extend(procedural_results)
            
            return "\n".join(results)
            
        return "No results found or invalid memory type."
    
    # Tool to store new information in memory
    def store_memory(
        memory_type: str = "semantic", 
        content: str = "", 
        category: str = "general"
    ) -> str:
        """
        Store new information in memory
        
        Args:
            memory_type: Type of memory to store (semantic, episodic, procedural)
            content: Content to store
            category: Category for the memory (for semantic: preference, knowledge, etc.)
            
        Returns:
            Confirmation message
        """
        if not content:
            return "Error: Content cannot be empty"
            
        if memory_type == "semantic":
            memory_system.store_semantic_memory({
                "content": content,
                "category": category
            })
            return f"Stored in semantic memory under category '{category}'"
            
        elif memory_type == "episodic":
            memory_system.store_episodic_memory({
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": {"category": category}
            })
            return f"Stored in episodic memory"
            
        elif memory_type == "procedural":
            if category not in ["tools", "workflows", "preferences", "response_patterns"]:
                return f"Invalid procedural category. Use: tools, workflows, preferences, response_patterns"
                
            memory_system.update_procedural_memory(
                category,
                f"manual_{int(time.time())}",
                {"description": content}
            )
            return f"Stored in procedural memory under {category}"
            
        return "Invalid memory type. Use: semantic, episodic, procedural"
    
    # Create Pydantic models for the tools
    class SearchMemoryInput(BaseModel):
        memory_type: str = Field(
            description="Type of memory to search (episodic, semantic, procedural, all)",
            default="all"
        )
        query: str = Field(
            description="Search query to find relevant memories"
        )
        limit: int = Field(
            description="Maximum number of results to return",
            default=5
        )
    
    class StoreMemoryInput(BaseModel):
        memory_type: str = Field(
            description="Type of memory to store (semantic, episodic, procedural)",
            default="semantic"
        )
        content: str = Field(
            description="Content to store in memory"
        )
        category: str = Field(
            description="Category for the memory (e.g., preference, knowledge, task)",
            default="general"
        )
    
    # Create the tools
    search_memory_tool = StructuredTool.from_function(
        func=search_memory,
        name="search_memory",
        description="Search the assistant's memory for relevant information",
        args_schema=SearchMemoryInput
    )
    
    store_memory_tool = StructuredTool.from_function(
        func=store_memory,
        name="store_memory",
        description="Store new information in the assistant's memory",
        args_schema=StoreMemoryInput
    )
    
    return [search_memory_tool, store_memory_tool]

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# Define state schema
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    memories: Optional[Dict[str, Any]] = None

# Initialize the memory system and LLM
memory_system = MemorySystem(user_id="default_user", embedding_model="local")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Initialize the background memory processor
memory_processor = BackgroundMemoryProcessor(memory_system, llm)

# Add memory tools to the agent tools
memory_tools = create_memory_tools(memory_system)
agent_tools = [
    *music_player_tools, 
    *notion_tools, 
    *gmail_tools, 
    *calendar_tools, 
    *twitter_tools, 
    web_search_tool, 
    procrastination_check_tool, 
    *memory_tools,
    *screen_tools,    # Add screen analysis tools 
    *keyboard_tools    # Add keyboard control tools
]
tool_node = ToolNode(agent_tools)

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that can control Spotify playback, manage tasks in Notion, and handle Gmail operations.
    You can:
    - Add new tasks to Notion with priorities and dates
    - Read existing tasks from Notion
    - Filter tasks by status
    - Show recent tasks
    - Get summaries of recent emails
    - Create draft emails
    - Add events to your calendar
    - View upcoming calendar events
    - Draft and post tweets to Twitter
    - Search the web for current information like weather, news, and real-time data
    - Monitor for procrastination and provide gentle, motivational reminders
    - Access and update your memory to provide more personalized assistance
    - Capture and analyze what's on the user's screen
    - Control the keyboard to type text or press keys
    
    Your memory system consists of:
    - Working memory: What you're focusing on right now
    - Episodic memory: Specific past interactions with the user
    - Semantic memory: General facts and user preferences
    - Procedural memory: Common patterns and workflows
    
    When using the memory system:
    - Search memory to recall relevant past interactions
    - Store important information you learn about the user
    - Use memory to provide consistent, personalized responses
    
    When the user is procrastinating:
    - Be supportive but firm in your reminders
    - Reference their specific tasks from Notion
    - Be sarcastic and make fun of them unfiltered and brutally honest
    - Make them feel bad and guilty
    When controlling the keyboard:
    - Type text carefully and accurately 
    - Use keyboard shortcuts appropriately
    - Confirm successful typing/key press operations
    
    When analyzing the screen:
    - Provide detailed descriptions of what you see
    - Focus on the specific information the user is asking about
    - Identify text, UI elements, and other visual content
   
    Always respond in a friendly, conversational manner, keep it short.
    When searching for current information, be specific in your queries to get the most accurate results.
    When adding tasks to Notion, make sure to set appropriate priorities and dates.
    When handling emails, be clear and professional.
    When adding calendar events, ensure the time format is correct (YYYY-MM-DDTHH:MM:SS).
    When handling tweets:
    - Keep them under 280 characters
    - Be professional and engaging
    - Only use hashtags if specifically requested
    - If a user wants to post a tweet, first draft it and ask for their approval"""),
    MessagesPlaceholder(variable_name="memory_context", optional=True),
    MessagesPlaceholder(variable_name="messages"),
])

# Add a Timer class for periodic checking
class PeriodicChecker:
    def __init__(self, interval=300):  # Default: check every 5 minutes
        self.interval = interval
        self.timer = None
        self.running = False
        self.last_check = datetime.now()
        
    def start(self, callback):
        """Start periodic checking"""
        self.running = True
        
        def run():
            if self.running:
                callback()
                self.last_check = datetime.now()
                self.timer = threading.Timer(self.interval, run)
                self.timer.daemon = True
                self.timer.start()
                
        # Start the first timer
        self.timer = threading.Timer(self.interval, run)
        self.timer.daemon = True
        self.timer.start()
        
    def stop(self):
        """Stop periodic checking"""
        self.running = False
        if self.timer:
            self.timer.cancel()

# Function to create agent state with memory processor
def create_agent_state(messages=None):
    """Create a new agent state with messages and memory processor"""
    return {
        "messages": messages or [],
        "memory_processor": memory_processor
    }

# Define the agent function
def agent(state: AgentState):
    messages = state["messages"]
    
    # Update working memory with current messages
    memory_system.update_working_memory(messages)
    
    # Get relevant memories based on the latest user message
    latest_user_message = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
    memory_context = []
    
    if latest_user_message:
        # Get relevant memories across all memory types
        memory_content = memory_system.format_for_prompt(
            memory_types=["working", "episodic", "semantic", "procedural"],
            query=latest_user_message,
            limit=3
        )
        
        if memory_content:
            memory_context = [SystemMessage(content=f"Memory context:\n{memory_content}")]
    
    # Use a more compatible approach with the LLM
    # Bind tools directly to the LLM with the prompt
    agent_runnable = prompt | llm.bind_tools(agent_tools)
    
    # Invoke the agent with the messages and memory context
    response = agent_runnable.invoke({
        "messages": messages,
        "memory_context": memory_context
    })
    
    # Queue conversation for background memory processing instead of immediate processing
    if hasattr(state, 'memory_processor') and state.get('memory_processor'):
        state['memory_processor'].add_conversation(messages + [response])
    else:
        # Fall back to immediate processing if no background processor
        extract_memory_insights(messages + [response], memory_system)
    
    return {"messages": messages + [response]}

# Define the function that determines whether to continue
def should_continue(state):
    last_message = state["messages"][-1]
    
    # Check direct tool_calls attribute
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "continue"
    
    # Also check additional_kwargs for tool_calls (alternative format)
    if hasattr(last_message, 'additional_kwargs') and 'tool_calls' in last_message.additional_kwargs:
        if last_message.additional_kwargs['tool_calls']:
            return "continue"
            
    return "end"

# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
        
    # Execute the tools and get responses
    responses = tool_node.invoke({"messages": messages})
    
    # Add tool responses to messages
    return {"messages": messages + responses["messages"]}

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent)
workflow.add_node("action", call_tool)

# Set the entry point
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)
workflow.add_edge("action", "agent")

# Compile the graph
app = workflow.compile()

# Create the chat interface function
def chat(message: str, chat_history: list[BaseMessage] = None) -> list[BaseMessage]:
    if chat_history is None:
        chat_history = []
    
    # Skip processing if message is empty
    if not message or message.strip() == "":
        print("Warning: Empty message received, skipping processing")
        return chat_history
    
    # Add user message to history
    messages = [*chat_history, HumanMessage(content=message)]
    
    # Create initial state with memory processor
    initial_state = create_agent_state(messages)
    
    # Run the graph
    result = app.invoke(initial_state)
    
    return result["messages"]

# Interactive chat interface with session management
def interactive_chat(session_id=None, max_messages=50):
    print("Assistant initialized. Type 'quit' to exit.")
    print("Special commands:")
    print("  /new - Start a new session")
    print("  /save - Save current session")
    print("  /sessions - List available sessions")
    print("  /load [session_id] - Load a specific session")
    print("  /memory - Show memory stats")
    print("  /memory_queue - Show memory queue status")
    print("  /help - Show available commands")
    
    # Set user_id based on session
    user_id = session_id if session_id else "default_user"
    
    # Declare all globals at the beginning of the function
    global memory_system, memory_tools, agent_tools, tool_node, memory_processor
    
    # Initialize or load the memory system for this user
    memory_system = MemorySystem(user_id=user_id)
    
    # Initialize background memory processor
    memory_processor = BackgroundMemoryProcessor(memory_system, llm)
    memory_processor.start()
    
    # Recreate memory tools with the new memory system
    memory_tools = create_memory_tools(memory_system)
    
    # Update agent tools - now including screen and keyboard tools
    agent_tools = [
        *music_player_tools, 
        *notion_tools, 
        *gmail_tools, 
        *calendar_tools, 
        *twitter_tools, 
        web_search_tool, 
        procrastination_check_tool, 
        *memory_tools,
        *screen_tools,      # Add screen analysis tools
        *keyboard_tools     # Add keyboard control tools
    ]
    
    # Update tool node
    tool_node = ToolNode(agent_tools)
    
    # Load previous chat history if session_id is provided
    chat_history = load_conversation(session_id) if session_id else []
    current_session_id = session_id
    
    # If we loaded a session, show a summary
    if chat_history:
        print(f"\nLoaded session with {len(chat_history)} messages.")
        # Update working memory with loaded history
        memory_system.update_working_memory(chat_history)
        # Optionally show the last few messages
        for msg in chat_history[-3:]:
            if isinstance(msg, HumanMessage):
                print(f"\nYou: {msg.content[:50]}{'...' if len(msg.content) > 50 else ''}")
            elif isinstance(msg, AIMessage):
                print(f"\nAssistant: {msg.content[:50]}{'...' if len(msg.content) > 50 else ''}")
        print("\n--- End of session history summary ---\n")
    
    # Initialize procrastination checker
    def check_procrastination():
        try:
            print("\n--- Procrastination check starting ---")
            notification = check_procrastination_notifications()
            print(f"Notification response: '{notification[:50]}{'...' if len(notification) > 50 else ''}'")
            
            # Skip if no notification or if it's a "no notifications" message
            if not notification or "No procrastination notifications found" in notification:
                print("No procrastination notifications to process")
                return
            
            print(f"\nProcrastination notification received: {notification[:100]}...")
            
            # Create a system message with the notification
            system_msg = SystemMessage(content=f"[Procrastination Alert]: {notification}")
            chat_history.append(system_msg)
            print("Added notification to chat history")
            
            # Have the agent respond to the procrastination notification
            print("\nSystem: Procrastination detected! Assistant is being notified...")
            
            # Use a specific prompt for the agent
            prompt = "The procrastination monitor detected that I'm off-task. Please give me a gentle reminder to focus on my tasks. Be motivational but firm. Reference my current tasks if possible."
            print(f"Sending prompt to agent: '{prompt[:50]}...'")
            
            try:
                print("Calling agent with prompt...")
                agent_response = chat(prompt, chat_history)
                print(f"Received response from agent with {len(agent_response)} messages")
                
                # Update chat history with the new messages
                chat_history.clear()
                chat_history.extend(agent_response)
                print("Updated chat history with agent response")
                
                # Print the assistant's response
                last_message = agent_response[-1] if agent_response else None
                if isinstance(last_message, AIMessage):
                    print("\nAssistant:", last_message.content)
                else:
                    print("\nNo response from assistant for procrastination notification")
            except Exception as e:
                print(f"\nError getting procrastination response: {str(e)}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f"\nError in procrastination checker: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Create and start the checker
    proc_checker = PeriodicChecker()
    proc_checker.start(check_procrastination)
    
    try:
        while True:
            user_input = input("\nYou: ")
            
            # Skip empty inputs
            if not user_input or user_input.strip() == "":
                print("Please enter a message or command.")
                continue
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                # Save conversation and memories before exiting
                if chat_history:
                    current_session_id = save_conversation(chat_history, current_session_id)
                    print(f"\nSession saved as: {current_session_id}")
                memory_system.save_memories()
                print("\nGoodbye!")
                break
            
            elif user_input.startswith("/new"):
                # Save current session and start a new one
                if chat_history:
                    current_session_id = save_conversation(chat_history, current_session_id)
                    print(f"\nPrevious session saved as: {current_session_id}")
                memory_system.save_memories()
                chat_history = []
                current_session_id = None
                # Create a new user ID for the memory system
                user_id = f"user_{int(time.time())}"
                memory_system = MemorySystem(user_id=user_id)
                print("\nStarted a new session with fresh memory.")
                continue
            
            elif user_input.startswith("/save"):
                # Save current session and memories
                if chat_history:
                    current_session_id = save_conversation(chat_history, current_session_id)
                    memory_system.save_memories()
                    print(f"\nSession and memories saved as: {current_session_id}")
                else:
                    print("\nNo messages to save.")
                continue
            
            elif user_input.startswith("/sessions"):
                # List available sessions
                sessions = list_conversation_sessions()
                if sessions:
                    print("\nAvailable sessions:")
                    for i, (sid, readable) in enumerate(sessions):
                        print(f"{i+1}. {readable} (ID: {sid})")
                else:
                    print("\nNo saved sessions found.")
                continue
            
            elif user_input.startswith("/memory"):
                # Show memory statistics
                episodic_count = len(memory_system.episodic_memory.index_to_docstore_id)
                semantic_count = len(memory_system.semantic_memory.index_to_docstore_id)
                procedural_count = sum(len(patterns) for patterns in memory_system.procedural_memory.values())
                
                print("\nMemory Statistics:")
                print(f"- Working memory: {len(memory_system.working_memory['recent_messages'])} recent messages")
                print(f"- Episodic memory: {episodic_count} memories")
                print(f"- Semantic memory: {semantic_count} facts")
                print(f"- Procedural memory: {procedural_count} patterns")
                
                # Show some sample memories if available
                if episodic_count > 1:  # Ignore initialization document
                    print("\nRecent episodic memories:")
                    docs = memory_system.search_episodic_memory("", k=3)
                    for i, doc in enumerate(docs):
                        print(f"  {i+1}. {doc.page_content[:100]}...")
                
                if semantic_count > 1:  # Ignore initialization document
                    print("\nSample semantic memories:")
                    docs = memory_system.search_semantic_memory("", k=3)
                    for i, doc in enumerate(docs):
                        print(f"  {i+1}. {doc.page_content[:100]}...")
                continue
            
            elif user_input.startswith("/memory_queue"):
                # Show memory queue status
                queue_size = memory_processor.queue.qsize()
                print(f"\nMemory processing queue: {queue_size} items waiting")
                continue
            
            elif user_input.startswith("/load"):
                # Load a specific session
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    try:
                        # Try loading by session ID
                        load_id = parts[1]
                        loaded_history = load_conversation(load_id)
                        if loaded_history:
                            # Save current session before loading new one
                            if chat_history:
                                save_conversation(chat_history, current_session_id)
                                memory_system.save_memories()
                            
                            chat_history = loaded_history
                            current_session_id = load_id
                            
                            # Load or create memory for this session
                            memory_system = MemorySystem(user_id=load_id)
                            
                            # Update memory tools with the new memory system
                            memory_tools = create_memory_tools(memory_system)
                            
                            # Update agent tools
                            agent_tools = [*music_player_tools, *notion_tools, *gmail_tools, *calendar_tools, *twitter_tools, web_search_tool, procrastination_check_tool, *memory_tools, *screen_tools, *keyboard_tools]
                            
                            # Update tool node
                            tool_node = ToolNode(agent_tools)
                            
                            # Update working memory with loaded history
                            memory_system.update_working_memory(chat_history)
                            
                            print(f"\nLoaded session with {len(chat_history)} messages.")
                        else:
                            print(f"\nSession '{load_id}' not found.")
                    except Exception as e:
                        print(f"\nError loading session: {str(e)}")
                else:
                    print("\nPlease specify a session ID to load.")
                continue
            
            elif user_input.startswith("/help"):
                print("\nAvailable commands:")
                print("  /new - Start a new session")
                print("  /save - Save current session")
                print("  /sessions - List available sessions")
                print("  /load [session_id] - Load a specific session")
                print("  /memory - Show memory statistics")
                print("  /memory_queue - Show memory queue status")
                print("  /help - Show this help message")
                print("  quit - Exit the application")
                continue
            
            # Limit conversation history size if needed
            if len(chat_history) > max_messages:
                # Keep important context: system message, and recent messages
                system_messages = [msg for msg in chat_history if isinstance(msg, SystemMessage)]
                recent_messages = chat_history[-max_messages:]
                chat_history = system_messages + recent_messages
                print("\nTrimmed conversation history to last", max_messages, "messages for efficiency.")
            
            try:
                # Process the user message
                chat_history = chat(user_input, chat_history)
                
                # Auto-save after every few messages
                if len(chat_history) % 10 == 0:  # Save every 10 messages
                    current_session_id = save_conversation(chat_history, current_session_id)
                    memory_system.save_memories()
                    print(f"\n(Session and memories auto-saved as: {current_session_id})")
                
                # Print the assistant's response
                last_message = chat_history[-1]
                if isinstance(last_message, AIMessage):
                    print("\nAssistant:", last_message.content)
                elif isinstance(last_message, ToolMessage):
                    print("\nSystem:", last_message.content)
                
            except Exception as e:
                print(f"\nError: {str(e)}")
                continue
    finally:
        # Stop the memory processor
        memory_processor.stop()
        
        # Stop the procrastination checker
        proc_checker.stop()
        
        # Save memories before exiting
        memory_system.save_memories()

# Function to save conversation history
def save_conversation(chat_history, session_id=None):
    """Save the current conversation history to a file."""
    if session_id is None:
        # Generate a timestamp-based session ID if none provided
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory if it doesn't exist
    os.makedirs("conversation_history", exist_ok=True)
    
    # Extract only serializable data from messages before saving
    serializable_history = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            serializable_history.append(HumanMessage(content=msg.content))
        elif isinstance(msg, AIMessage):
            serializable_history.append(AIMessage(content=msg.content))
        elif isinstance(msg, SystemMessage):
            serializable_history.append(SystemMessage(content=msg.content))
        elif isinstance(msg, ToolMessage):
            # Add tool_call_id parameter for compatibility with newer LangChain versions
            serializable_history.append(ToolMessage(
                content=msg.content,
                name=msg.name,
                tool_call_id=getattr(msg, "tool_call_id", f"tool_{uuid.uuid4()}")
            ))
    # Save the clean history
    with open(f"conversation_history/session_{session_id}.pkl", "wb") as f:
        pickle.dump(serializable_history, f)
    
    return session_id


# Function to load conversation history
def load_conversation(session_id=None):
    """Load a conversation history from a file."""
    # If no session_id is provided, load the most recent session
    if session_id is None:
        history_files = sorted(os.listdir("conversation_history"))
        if not history_files:
            return []  # No history found
        session_id = history_files[-1].replace("session_", "").replace(".pkl", "")
    
    try:
        with open(f"conversation_history/session_{session_id}.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []  # Return empty history if file not found

# Function to list available conversation sessions
def list_conversation_sessions():
    """List all available conversation sessions."""
    if not os.path.exists("conversation_history"):
        return []
    
    sessions = []
    for filename in os.listdir("conversation_history"):
        if filename.startswith("session_") and filename.endswith(".pkl"):
            session_id = filename.replace("session_", "").replace(".pkl", "")
            # For timestamp-based IDs, convert to readable format
            if session_id.isdigit() or "_" in session_id:
                try:
                    # Try to format as a datetime if it's a timestamp-based ID
                    dt = datetime.strptime(session_id, "%Y%m%d_%H%M%S")
                    sessions.append((session_id, dt.strftime("%Y-%m-%d %H:%M:%S")))
                except ValueError:
                    sessions.append((session_id, session_id))
            else:
                sessions.append((session_id, session_id))
    
    return sessions

# Add a voice chat interface function
def voice_chat_interface():
    print("Voice Assistant initialized. Press and hold 'right shift' to speak. Press 'esc' to exit.")
    print("Using Kokoro TTS for speech synthesis and Whisper for speech recognition.")
    print("You can:")
    print("- Play songs by name or lyrics with Spotify")
    print("- Control playback (play, pause, next, previous)")
    print("- Manage tasks in Notion")
    print("- Handle emails with Gmail")
    print("- Manage calendar events")
    print("- Search the web for information")
    print("- Post tweets")
    print("- Analyze what's currently on your screen")
    print("- Control the keyboard to type text or press keys")
    
    # Set user_id for the session
    user_id = f"voice_user_{int(time.time())}"
    
    # Initialize memory system for this user
    memory_system = MemorySystem(user_id=user_id)
    
    # Initialize background memory processor
    memory_processor = BackgroundMemoryProcessor(memory_system, llm)
    memory_processor.start()
    
    # Recreate memory tools with the new memory system
    memory_tools = create_memory_tools(memory_system)
    
    # Update agent tools with screen and keyboard tools
    agent_tools = [
        *music_player_tools, 
        *notion_tools, 
        *gmail_tools, 
        *calendar_tools, 
        *twitter_tools, 
        web_search_tool, 
        procrastination_check_tool, 
        *memory_tools,
        *screen_tools,      # Add screen analysis tools
        *keyboard_tools     # Add keyboard control tools
    ]
    
    # Update tool node
    tool_node = ToolNode(agent_tools)
    
    chat_history = []
    running = True
    
    def check_exit():
        nonlocal running
        while running:
            if keyboard.is_pressed('esc'):
                print("\nExiting voice assistant...")
                running = False
            sleep(0.1)
    
    # Start exit checker in separate thread
    exit_thread = threading.Thread(target=check_exit)
    exit_thread.daemon = True
    exit_thread.start()
    
    # Define procrastination check function
    def check_procrastination():
        try:
            print("\n--- Procrastination check starting ---")
            notification = check_procrastination_notifications()
            print(f"Notification response: '{notification[:50]}{'...' if len(notification) > 50 else ''}'")
            
            # Skip if no notification or if it's a "no notifications" message
            if not notification or "No procrastination notifications found" in notification:
                print("No procrastination notifications to process")
                return
            
            print(f"\nProcrastination notification received: {notification[:100]}...")
            
            # Create a system message with the notification
            system_msg = SystemMessage(content=f"[Procrastination Alert]: {notification}")
            chat_history.append(system_msg)
            print("Added notification to chat history")
            
            # Have the agent respond to the procrastination notification
            print("\nSystem: Procrastination detected! Assistant is being notified...")
            
            # Use a specific prompt for the agent
            prompt = "The procrastination monitor detected that I'm off-task. Please give me a gentle reminder to focus on my tasks. Be motivational but firm. Reference my current tasks if possible."
            print(f"Sending prompt to agent: '{prompt[:50]}...'")
            
            try:
                print("Calling agent with prompt...")
                # Create initial state with memory processor
                messages = [*chat_history, HumanMessage(content=prompt)]
                initial_state = create_agent_state(messages)
                
                # Run the graph
                result = app.invoke(initial_state)
                chat_history.clear()
                chat_history.extend(result["messages"])
                print(f"Received response from agent with {len(chat_history)} messages")
                
                # Print and speak the assistant's response
                last_message = chat_history[-1] if chat_history else None
                if isinstance(last_message, AIMessage):
                    response_text = last_message.content
                    print("\nAssistant:", response_text)
                    speak_response(response_text)
                else:
                    print("\nNo response from assistant for procrastination notification")
            except Exception as e:
                print(f"\nError getting procrastination response: {str(e)}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f"\nError in procrastination checker: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Initialize procrastination checker
    proc_checker = PeriodicChecker()
    proc_checker.start(check_procrastination)
    
    try:
        while running:
            if keyboard.is_pressed('right shift'):
                user_input = listen_while_pressed()
                
                if user_input:
                    print("\nYou said:", user_input)
                    
                    try:
                        # Create initial state with memory processor
                        messages = [*chat_history, HumanMessage(content=user_input)]
                        initial_state = create_agent_state(messages)
                        
                        # Run the graph
                        result = app.invoke(initial_state)
                        chat_history = result["messages"]
                        
                        # Get the last message
                        last_message = chat_history[-1]
                        if isinstance(last_message, AIMessage):
                            response_text = last_message.content
                            print("\nAssistant:", response_text)
                            speak_response(response_text)
                        elif isinstance(last_message, ToolMessage):
                            response_text = last_message.content
                            print("\nSystem:", response_text)
                            speak_response(response_text)
                            
                    except Exception as e:
                        error_message = f"Sorry, there was an error: {str(e)}"
                        print("\nError:", error_message)
                        speak_response(error_message)
            
            sleep(0.1)  # Prevent high CPU usage
            
    finally:
        # Stop the memory processor
        memory_processor.stop()
        
        # Stop the procrastination checker
        proc_checker.stop()
        
        # Save memories before exiting
        memory_system.save_memories()

# Define speech-to-text functions
def listen_for_speech():
    """Function to capture audio and convert to text using Groq's Whisper"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\nListening...")
        try:
            audio = recognizer.listen(source, timeout=5)
            print("Processing speech...")
            
            # Convert audio data to format suitable for Groq
            with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as temp_audio:
                # Save audio to temporary file
                temp_audio.write(audio.get_wav_data())
                temp_audio.flush()
                
                # Use Groq to transcribe
                with open(temp_audio.name, "rb") as audio_file:
                    transcription = groq_client.audio.transcriptions.create(
                        file=(temp_audio.name, audio_file.read()),
                        model="whisper-large-v3-turbo",
                        response_format="json",
                        language="en",
                        temperature=0.0
                    )
            
            # Clean up temp file
            os.unlink(temp_audio.name)
            return transcription.text
            
        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            print(f"Error in speech recognition: {str(e)}")
            return None

def listen_while_pressed(key='right shift'):
    """Function to capture audio while a key is being held down"""
    recognizer = sr.Recognizer()
    
    # Optimize recognition settings
    recognizer.energy_threshold = 1000  # Increase sensitivity
    recognizer.dynamic_energy_threshold = False  # Disable dynamic adjustment
    recognizer.pause_threshold = 0.5  # Reduce pause time
    recognizer.phrase_threshold = 0.3  # Lower phrase time
    
    with sr.Microphone() as source:
        print("\nListening...")
        try:
            # Wait for audio while key is pressed
            audio = recognizer.listen(source, timeout=None, phrase_time_limit=None)
            print("Processing speech...")
            
            # Use a temporary file in memory instead of disk when possible
            with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as temp_audio:
                temp_audio.write(audio.get_wav_data())
                temp_audio.flush()
                
                with open(temp_audio.name, "rb") as audio_file:
                    transcription = groq_client.audio.transcriptions.create(
                        file=(temp_audio.name, audio_file.read()),
                        model="whisper-large-v3-turbo",  # Using turbo model for speed
                        response_format="json",
                        language="en",
                        temperature=0.0,
                        prompt="Convert speech to text"  # Help guide the model
                    )
            
            os.unlink(temp_audio.name)
            return transcription.text
            
        except Exception as e:
            print(f"Error in speech recognition: {str(e)}")
            return None

if __name__ == "__main__":
    # Ask if user wants to load previous session
    print("Welcome to Assistant!")
    print("Choose an interface:")
    print("1. Text-based chat (with memory)")
    print("2. Voice chat (press and hold right shift to talk)")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "2":
        voice_chat_interface()
    else:
        sessions = list_conversation_sessions()
        
        if sessions:
            print("\nFound previous sessions:")
            for i, (sid, readable) in enumerate(sessions[-5:]):  # Show last 5 sessions
                print(f"{i+1}. {readable} (ID: {sid})")
            
            choice = input("\nLoad most recent session? (y/n): ")
            if choice.lower() in ['y', 'yes']:
                # Load the most recent session
                session_id = sessions[-1][0]
                interactive_chat(session_id)
            else:
                # Start a new session
                interactive_chat()
        else:
            # No previous sessions found
            print("No previous sessions found. Starting a new session.")
            interactive_chat()