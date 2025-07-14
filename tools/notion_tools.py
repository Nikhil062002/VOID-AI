from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import os
import requests
from langchain.tools import tool
from pydantic import ValidationError
import logging

logger = logging.getLogger(__name__)

def handle_notion_errors(response: requests.Response) -> str:
    """Handle Notion API errors gracefully"""
    if 200 <= response.status_code < 300:
        return ""
        
    error_msg = f"Notion API Error ({response.status_code}): "
    try:
        error_data = response.json()
        error_msg += error_data.get("message", "Unknown error")
    except:
        error_msg += response.text
        
    logger.error(error_msg)
    return error_msg


class NotionTask(BaseModel):
    task: str = Field(..., description="The task description")
    priority: str = Field(
        default="medium", 
        description="Priority level of the task (low, medium, high)"
    )
    status: str = Field(
        default="Not started", 
        description="Status of the task (Not started, In Progress, Done)"
    )
    date: Optional[str] = Field(
        default=None, 
        description="Due date for the task in YYYY-MM-DD format"
    )

class ReadNotionParams(BaseModel):
    num_tasks: Optional[int] = Field(
        default=5,
        description="Number of tasks to retrieve (default: 5)",
        type="integer"
    )
    status: Optional[str] = Field(
        default=None,
        description="Filter tasks by status (Done, Not Started, In Progress)",
        type="string"
    )    

class TaskProgress(BaseModel):
    task_id: str = Field(..., description="Notion task ID")
    time_spent: int = Field(..., description="Time spent in minutes")
    productive: bool = Field(..., description="Was the time productive?")



@tool("add_notion_task", return_direct=True, args_schema=NotionTask)
def add_notion_task(
    task: str,
    priority: str = "medium",
    status: str = "Not started",
    date: Optional[str] = None,
) -> str:
    """
    Add a new task to Notion database.
    
    Args:
        task: The task description
        priority: Priority level (low, medium, high)
        status: Task status (Not started, In progress, Done)
        date: Due date in YYYY-MM-DD format
    
    Returns:
        str: Confirmation message
    """
    DATABASE_ID = os.getenv("DATABASE_ID")
    NOTION_API_KEY = os.getenv("NOTION_API_KEY")
    
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }
    
    # Use current date if none provided
    if not date:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    data = {
        "parent": {"database_id": DATABASE_ID},
        "properties": {
            "s": {"status": {"name": status}},
            "Date": {"date": {"start": date}},
            "priority": {"select": {"name": priority.lower()}},
            "task": {"title": [{"text": {"content": task}}]}
        }
    }
    
    response = requests.post(
        "https://api.notion.com/v1/pages",
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        return f"Successfully added task: {task}"
    else:
        return f"Failed to add task. Error: {response.text}"


@tool("read_notion_tasks", return_direct=True, args_schema=ReadNotionParams)
def read_notion_tasks(
    num_tasks: Optional[int] = 5,
    status: Optional[str] = None
) -> str:
    """
    Read tasks from Notion database.
    
    Args:
        num_tasks: Number of tasks to retrieve (default: 5)
        status: Filter tasks by status (optional)
    
    Returns:
        str: Conversational response about tasks
    """
    DATABASE_ID = os.getenv("DATABASE_ID")
    NOTION_API_KEY = os.getenv("NOTION_API_KEY")
    
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }
    
    url = f"https://api.notion.com/v1/databases/{DATABASE_ID}/query"
    
    filter_params = {}
    if status:
        filter_params = {
            "filter": {
                "property": "s",
                "status": {
                    "equals": status
                }
            }
        }
    
    payload = {
        "page_size": num_tasks,
        **filter_params
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        return "I'm having trouble accessing your tasks right now."
    
    results = response.json()["results"]
    
    if not results:
        return "You don't have any tasks at the moment."
    
    # Create a conversational response
    status_msg = f" with status {status}" if status else ""
    intro = f"Here are your {len(results)} most recent tasks{status_msg}. "
    
    tasks = []
    for page in results:
        props = page["properties"]
        task_name = props["task"]["title"][0]["text"]["content"]
        status = props["s"]["status"]["name"]
        priority = props["priority"]["select"]["name"]
        date = props["Date"]["date"]["start"]
        
        # Convert date to more natural format
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            date_str = date_obj.strftime("%B %d")
        except:
            date_str = date
            
        task_str = f"You have a {priority} priority task to {task_name}, due on {date_str}. Its status is {status}."
        tasks.append(task_str)
    
    return intro + " ".join(tasks)


def read_tasks(
    num_tasks: Optional[int] = 5,
    status: Optional[str] = None
) -> str:
    """
    Read tasks from Notion database.
    
    Args:
        num_tasks: Number of tasks to retrieve (default: 5)
        status: Filter tasks by status (optional)
    
    Returns:
        str: Conversational response about tasks
    """
    DATABASE_ID = os.getenv("DATABASE_ID")
    NOTION_API_KEY = os.getenv("NOTION_API_KEY")
    
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }
    
    url = f"https://api.notion.com/v1/databases/{DATABASE_ID}/query"
    
    filter_params = {}
    if status:
        filter_params = {
            "filter": {
                "property": "s",
                "status": {
                    "equals": status
                }
            }
        }
    
    payload = {
        "page_size": num_tasks,
        **filter_params
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        return "I'm having trouble accessing your tasks right now."
    
    results = response.json()["results"]
    
    if not results:
        return "You don't have any tasks at the moment."
    
    # Create a conversational response
    status_msg = f" with status {status}" if status else ""
    #intro = f"Here are your {len(results)} most recent tasks{status_msg}. "
    
    tasks = []
    for page in results:
        props = page["properties"]
        task_name = props["task"]["title"][0]["text"]["content"]
        status = props["s"]["status"]["name"]
        priority = props["priority"]["select"]["name"]
        date = props["Date"]["date"]["start"]
        
        # Convert date to more natural format
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            date_str = date_obj.strftime("%B %d")
        except:
            date_str = date
            
        task_str = f"{task_name}."
        tasks.append(task_str)
    
    return " ".join(tasks)

# Update the tools list to include both functions
notion_tools = [add_notion_task, read_notion_tasks]



