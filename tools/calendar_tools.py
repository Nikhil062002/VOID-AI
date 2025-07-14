from datetime import datetime, timedelta
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os.path
import pickle
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
# If modifying these scopes, delete the file token.pickle.
# Update the SCOPES at the top of the file
SCOPES = [
    'https://www.googleapis.com/auth/calendar',  # Calendar access
    'https://www.googleapis.com/auth/gmail.readonly',  # Read Gmail
    'https://www.googleapis.com/auth/gmail.compose'    # Create Gmail drafts
]
class AddEventInput(BaseModel):
    title: str = Field(..., description="Title of the event")
    start_time: str = Field(..., description="Start time of the event in ISO format (YYYY-MM-DDTHH:MM:SS)")
    end_time: Optional[str] = Field(None, description="Optional end time of the event in ISO format (YYYY-MM-DDTHH:MM:SS)", type="string")
    description: Optional[str] = Field(None, description="Optional description of the event", type="string")

class GetEventsInput(BaseModel):
    max_results: int = Field(
        default=10,
        description="Maximum number of events to return"
    )
def get_calendar_service():
    """Gets or refreshes Calendar API service."""
    creds = None
    # The file token.pickle stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('calendar', 'v3', credentials=creds)

def add_calendar_event(
    title: str,
    start_time: str,
    end_time: Optional[str] = None,
    description: Optional[str] = None
) -> str:
    """
    Add an event to Google Calendar.
    """
    try:
        service = get_calendar_service()
        
        # Parse start time
        start_datetime = datetime.fromisoformat(start_time)
        
        # If no end time provided, set to 1 hour after start
        if not end_time:
            end_datetime = start_datetime + timedelta(hours=1)
        else:
            end_datetime = datetime.fromisoformat(end_time)

        # Print debug information
        print(f"Creating event:")
        print(f"Title: {title}")
        print(f"Start: {start_datetime}")
        print(f"End: {end_datetime}")

        event = {
            'summary': title,
            'description': description,
            'start': {
                'dateTime': start_datetime.isoformat(),
                'timeZone': 'America/New_York',  # Change this to your timezone
            },
            'end': {
                'dateTime': end_datetime.isoformat(),
                'timeZone': 'America/New_York',  # Change this to your timezone
            },
        }

        event = service.events().insert(calendarId='primary', body=event).execute()
        return f"Event created successfully: {title} (Event ID: {event.get('id')})"
    
    except ValueError as e:
        return f"Invalid date format: {str(e)}"
    except Exception as e:
        return f"Failed to create event: {str(e)}\nEvent data: {event}"

def add_calendar_event(
    title: str,
    start_time: str,
    end_time: Optional[str] = None,
    description: Optional[str] = None
) -> str:
    """
    Add an event to Google Calendar.
    
    Args:
        title: Event title
        start_time: Start time in ISO format or natural language
        end_time: Optional end time (defaults to 1 hour after start)
        description: Optional event description
    
    Returns:
        str: Confirmation message
    """
    try:
        service = get_calendar_service()
        
        # Parse start time
        start_datetime = datetime.fromisoformat(start_time)
        
        # If no end time provided, set to 1 hour after start
        if not end_time:
            end_datetime = start_datetime + timedelta(hours=1)
        else:
            end_datetime = datetime.fromisoformat(end_time)

        event = {
            'summary': title,
            'description': description,
            'start': {
                'dateTime': start_datetime.isoformat(),
                'timeZone': 'UTC',
            },
            'end': {
                'dateTime': end_datetime.isoformat(),
                'timeZone': 'UTC',
            },
        }

        event = service.events().insert(calendarId='primary', body=event).execute()
        return f"Event created successfully: {title}"
    
    except Exception as e:
        return f"Failed to create event: {str(e)}"

def get_upcoming_events(max_results: int = 10) -> str:
    """
    Get upcoming calendar events.
    
    Args:
        max_results: Maximum number of events to return
    
    Returns:
        str: Formatted list of upcoming events
    """
    try:
        service = get_calendar_service()
        
        # Get current time in UTC
        now = datetime.utcnow().isoformat() + 'Z'
        
        # Call the Calendar API
        events_result = service.events().list(
            calendarId='primary',
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])

        if not events:
            return "No upcoming events found."
            
        # Format the events
        output = "Upcoming events:\n"
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            start_time = datetime.fromisoformat(start.replace('Z', '+00:00'))
            output += f"- {event['summary']} at {start_time.strftime('%Y-%m-%d %H:%M')}\n"
            
        return output
        
    except Exception as e:
        return f"Failed to retrieve events: {str(e)}"

# List of tools to be used by the agent
calendar_tools = [
    StructuredTool(
        name="add_calendar_event",
        description="Add an event to Google Calendar",
        func=add_calendar_event,
        args_schema=AddEventInput
    ),
    StructuredTool(
        name="get_upcoming_events",
        description="Get a list of upcoming calendar events",
        func=get_upcoming_events,
        args_schema=GetEventsInput
    )
]