from pathlib import Path
from datetime import datetime
import json
from pydantic import BaseModel
from langchain.tools import StructuredTool
from filelock import FileLock

# Create a procrastination notification tool

def check_procrastination_notifications():
    """Check for notifications from the procrastination monitor"""
    print("\n--- Checking for procrastination notifications ---")
    NOTIFICATION_FILE = Path("notification_queue.json")
    LOCK_FILE = Path("notification_queue.json.lock")
    
    # If the file doesn't exist, create an empty file to prevent errors
    if not NOTIFICATION_FILE.exists():
        print("Notification file doesn't exist, creating it")
        try:
            with open(NOTIFICATION_FILE, 'w') as f:
                json.dump([], f)
        except Exception as e:
            print(f"Error creating notification file: {e}")
        return "No procrastination notifications found."
    else:
        print(f"Notification file exists at {NOTIFICATION_FILE}")
    
    try:
        notifications = []
        print(f"Acquiring file lock: {LOCK_FILE}")
        with FileLock(LOCK_FILE, timeout=10):  # Add timeout to prevent deadlocks
            try:
                print("Reading notification file")
                with open(NOTIFICATION_FILE, 'r') as f:
                    notifications = json.load(f)
                
                print(f"Found {len(notifications)} notifications in file")
                if not notifications:
                    print("Notification file exists but is empty")
                    return "No procrastination notifications found."
                
                # Get the most recent notification
                notification = notifications[-1]
                print(f"Processing notification: {notification.get('message', 'No message')}")
                
                # Mark it as read by removing it
                notifications.pop()
                
                # Save the updated notifications list
                print(f"Saving {len(notifications)} remaining notifications")
                with open(NOTIFICATION_FILE, 'w') as f:
                    json.dump(notifications, f, indent=2)
                    
                print(f"Processed notification, {len(notifications)} remain in queue")
            except json.JSONDecodeError:
                # Handle corrupted JSON file
                print("Notification file contains invalid JSON, resetting it")
                with open(NOTIFICATION_FILE, 'w') as f:
                    json.dump([], f)
                return "Error: Notification file was corrupted and has been reset."
            except Exception as e:
                print(f"Error processing notification file: {e}")
                return f"Error checking procrastination notifications: {str(e)}"
        
        # Only proceed if we actually got a notification
        if 'notification' not in locals():
            print("No valid notification found")
            return "No procrastination notifications found."
            
        # Format notification for the agent
        timestamp = datetime.fromisoformat(notification["timestamp"])
        time_ago = (datetime.now() - timestamp).total_seconds() / 60
        
        message = f"Procrastination detected {int(time_ago)} minutes ago. "
        message += f"The user was: {notification['activity'].get('activity', 'off-task')}. "
        
        if notification.get('current_tasks'):
            message += f"Current tasks: {', '.join(notification['current_tasks'][:3])}"
            if len(notification['current_tasks']) > 3:
                message += f" and {len(notification['current_tasks']) - 3} more."
        
        print(f"Returning notification message: {message[:50]}...")
        return message
    except Exception as e:
        print(f"Unexpected error in check_procrastination_notifications: {e}")
        return f"Error checking procrastination notifications: {str(e)}"

# Create a Pydantic model for the tool
class ProcrastinationCheckInput(BaseModel):
    """Input for checking procrastination notifications."""
    pass  # No inputs needed

# Create the structured tool
procrastination_check_tool = StructuredTool.from_function(
    func=check_procrastination_notifications,
    name="check_procrastination_notifications",
    description="Check if the procrastination monitor has detected any off-task behavior.",
    args_schema=ProcrastinationCheckInput,
    return_direct=False
)