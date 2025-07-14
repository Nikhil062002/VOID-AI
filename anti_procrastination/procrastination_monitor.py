import os
import time
import json
import schedule
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
import mss
import mss.tools
from PIL import Image
import google.generativeai as genai
from plyer import notification
from tools.notion_tools import read_tasks
import queue
import threading
from filelock import FileLock

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Constants
SCREENSHOT_INTERVAL = int(os.getenv("SCREENSHOT_INTERVAL", "10"))  # minutes
SCREENSHOT_DIR = Path("screenshots")
LOGS_DIR = Path("logs")
SUMMARY_FILE = LOGS_DIR / "activity_summary.json"

# Create directories if they don't exist
SCREENSHOT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Initialize Gemini model - use the correct model name
model = genai.GenerativeModel('gemini-2.0-flash')

# Create a global queue or file-based notification system
NOTIFICATION_FILE = Path("notification_queue.json")

class ProcrastinationMonitor:
    def __init__(self):
        self.last_notification_time = datetime.now() - timedelta(hours=1)
        self.current_tasks = []
        self.activity_log = []
        self.update_tasks_from_notion()
    
    def update_tasks_from_notion(self):
        """Fetch current tasks from Notion"""
        try:
            # Call read_notion_tasks with an empty string as input
            tasks_str = read_tasks()
            
            # Parse the tasks string into a list
            if isinstance(tasks_str, str):
                # Split by periods and clean up each task
                self.current_tasks = [task.strip() for task in tasks_str.split('.') if task.strip()]
            else:
                self.current_tasks = []
                
            print(f"Updated tasks: {len(self.current_tasks)} tasks found")
            print("Tasks:", self.current_tasks)  # Debug print
        except Exception as e:
            print(f"Error fetching tasks: {e}")
            self.current_tasks = []
    
    def take_screenshot(self) -> Optional[Path]:
        """Take a screenshot and save it to disk"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = SCREENSHOT_DIR / f"screen_{timestamp}.png"
            
            with mss.mss() as sct:
                # Capture the main monitor
                monitor = sct.monitors[1]
                sct_img = sct.grab(monitor)
                mss.tools.to_png(sct_img.rgb, sct_img.size, output=str(filename))
            
            print(f"Screenshot saved: {filename}")
            return filename
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            return None
    
    def analyze_screenshot(self, screenshot_path: Path) -> Dict:
        """Analyze screenshot using Gemini Vision model and check against tasks"""
        try:
            img = Image.open(screenshot_path)
            
            # Create a task list string to include in the prompt
            task_list = "\n".join([f"- {task}" for task in self.current_tasks])
            
            # Generate prompt for Gemini that includes task comparison
            prompt = f"""
            Analyze this screenshot and tell me:
            1. What application or website is open?
            2. What activity is being performed?
            3. Is this likely work-related or personal/entertainment?
            4. Does this activity match any of the following tasks? If yes, which one(s)?
            
            Tasks:
            {task_list}
            
            Format your response as JSON with keys:
            - 'application': string (what app/site is being used)
            - 'activity': string (description of what's happening)
            - 'is_work_related': boolean
            - 'matching_tasks': array of strings (which tasks match, if any)
            - 'on_task': boolean (is the user working on one of their listed tasks)
            """
            
            response = model.generate_content([prompt, img])
            
            # Extract JSON from response
            response_text = response.text
            # Find JSON in the response (it might be wrapped in markdown code blocks)
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].strip()
            else:
                json_str = response_text
            
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                result = {
                    "application": "Unknown",
                    "activity": response_text[:100],
                    "is_work_related": None,
                    "matching_tasks": [],
                    "on_task": False
                }
            
            print(f"Analysis result: {result}")
            return result
        except Exception as e:
            print(f"Error analyzing screenshot: {e}")
            return {
                "application": "Error",
                "activity": f"Analysis failed: {str(e)}",
                "is_work_related": None,
                "matching_tasks": [],
                "on_task": False
            }
    
    def send_notification(self, message: str):
        """Send a desktop notification and notify the main agent"""
        current_time = datetime.now()
        # Reduced the cooldown period for testing purposes
        notification_cooldown = 60  # Changed from 1800 seconds (30 min) to just 60 seconds (1 min)
        time_since_last = (current_time - self.last_notification_time).total_seconds()
        
        print(f"Time since last notification: {time_since_last:.1f} seconds (cooldown: {notification_cooldown} seconds)")
        
        # Only send notification if cooldown period has passed
        if time_since_last > notification_cooldown:
            try:
                print(f"Cooldown period passed, preparing to send notification: {message}")
                # Send desktop notification
                notification.notify(
                    title="Procrastination Monitor",
                    message=message,
                    timeout=10
                )
                
                # Also save notification for the main agent
                notification_data = {
                    "timestamp": current_time.isoformat(),
                    "message": message,
                    "activity": self.activity_log[-1] if self.activity_log else {},
                    "current_tasks": self.current_tasks
                }
                
                # Write to shared file that main agent can read
                try:
                    LOCK_FILE = str(NOTIFICATION_FILE) + ".lock"
                    
                    # Ensure the notification file exists
                    if not NOTIFICATION_FILE.exists():
                        try:
                            # Create parent directories if they don't exist
                            NOTIFICATION_FILE.parent.mkdir(exist_ok=True)
                            # Create empty file
                            with open(NOTIFICATION_FILE, 'w') as f:
                                json.dump([], f)
                        except Exception as e:
                            print(f"Error creating notification file: {e}")
                    
                    with FileLock(LOCK_FILE, timeout=10):  # Add timeout to prevent deadlocks
                        try:
                            # Read current notifications
                            if NOTIFICATION_FILE.exists():
                                with open(NOTIFICATION_FILE, 'r') as f:
                                    try:
                                        notifications = json.load(f)
                                    except json.JSONDecodeError:
                                        # Handle corrupted file
                                        print("Notification file corrupted, resetting")
                                        notifications = []
                            else:
                                notifications = []
                            
                            # Append new notification
                            notifications.append(notification_data)
                            
                            # Write updated notifications
                            with open(NOTIFICATION_FILE, 'w') as f:
                                json.dump(notifications, f, indent=2)
                            
                            print(f"Notification queued for main agent: {message}")
                            print(f"Queue now contains {len(notifications)} notifications")
                        except Exception as e:
                            print(f"Error processing notification file: {e}")
                except Exception as e:
                    print(f"Error saving notification for main agent: {e}")
                    
                self.last_notification_time = current_time
                print(f"Notification sent: {message}")
            except Exception as e:
                print(f"Error sending notification: {e}")
    
    def log_activity(self, timestamp: datetime, analysis: Dict, on_task: bool):
        """Log the activity for later summary"""
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "application": analysis.get("application"),
            "activity": analysis.get("activity"),
            "is_work_related": analysis.get("is_work_related"),
            "on_task": on_task,
            "matching_tasks": analysis.get("matching_tasks", [])
        }
        
        self.activity_log.append(log_entry)
        
        # Save to file
        try:
            SUMMARY_LOCK_FILE = str(SUMMARY_FILE) + ".lock"
            with FileLock(SUMMARY_LOCK_FILE):
                if SUMMARY_FILE.exists():
                    with open(SUMMARY_FILE, 'r') as f:
                        existing_log = json.load(f)
                else:
                    existing_log = []
                
                existing_log.append(log_entry)
                
                with open(SUMMARY_FILE, 'w') as f:
                    json.dump(existing_log, f, indent=2)
        except Exception as e:
            print(f"Error saving activity log: {e}")
    
    def generate_daily_summary(self):
        """Generate a summary of the day's activities"""
        try:
            if not SUMMARY_FILE.exists():
                print("No activity log found")
                return
            
            SUMMARY_LOCK_FILE = str(SUMMARY_FILE) + ".lock"
            with FileLock(SUMMARY_LOCK_FILE):
                with open(SUMMARY_FILE, 'r') as f:
                    log = json.load(f)
            
            # Filter for today's activities
            today = datetime.now().date()
            today_log = [
                entry for entry in log 
                if datetime.fromisoformat(entry["timestamp"]).date() == today
            ]
            
            if not today_log:
                print("No activities logged today")
                return
            
            # Calculate statistics
            total_entries = len(today_log)
            work_related = sum(1 for entry in today_log if entry.get("is_work_related"))
            on_task = sum(1 for entry in today_log if entry.get("on_task"))
            
            productivity_rate = (on_task / total_entries) * 100 if total_entries > 0 else 0
            
            # Group by application
            apps = {}
            for entry in today_log:
                app = entry.get("application", "Unknown")
                if app in apps:
                    apps[app] += 1
                else:
                    apps[app] = 1
            
            # Group by tasks
            tasks = {}
            for entry in today_log:
                for task in entry.get("matching_tasks", []):
                    if task in tasks:
                        tasks[task] += 1
                    else:
                        tasks[task] = 1
            
            # Sort apps and tasks by usage
            sorted_apps = sorted(apps.items(), key=lambda x: x[1], reverse=True)
            sorted_tasks = sorted(tasks.items(), key=lambda x: x[1], reverse=True)
            
            # Create summary message
            summary = f"Daily Summary ({today.strftime('%Y-%m-%d')}):\n"
            summary += f"Productivity rate: {productivity_rate:.1f}%\n"
            summary += f"Work-related activities: {work_related}/{total_entries}\n"
            summary += f"On-task activities: {on_task}/{total_entries}\n\n"
            
            summary += "Top applications used:\n"
            for app, count in sorted_apps[:5]:
                percentage = (count / total_entries) * 100
                summary += f"- {app}: {percentage:.1f}%\n"
            
            if sorted_tasks:
                summary += "\nTime spent on tasks:\n"
                for task, count in sorted_tasks:
                    minutes = count * SCREENSHOT_INTERVAL
                    summary += f"- {task}: ~{minutes} minutes\n"
            
            print(summary)
            
            # Send notification with summary
            self.send_notification(f"Daily summary ready. Productivity: {productivity_rate:.1f}%")
            
            # Save summary to file
            summary_path = LOGS_DIR / f"summary_{today.strftime('%Y%m%d')}.txt"
            with open(summary_path, 'w') as f:
                f.write(summary)
            
            print(f"Summary saved to {summary_path}")
        except Exception as e:
            print(f"Error generating summary: {e}")
    
    def monitor_cycle(self):
        """Run one monitoring cycle"""
        print(f"\n--- Monitoring cycle started at {datetime.now()} ---")
        
        # Update tasks from Notion
        self.update_tasks_from_notion()
        
        # Take screenshot
        screenshot_path = self.take_screenshot()
        if not screenshot_path:
            return
        
        # Analyze screenshot and check against tasks (now combined)
        analysis = self.analyze_screenshot(screenshot_path)
        
        # Log activity
        self.log_activity(datetime.now(), analysis, analysis.get("on_task", False))
        
        # For testing - more aggressive notification checking
        is_on_task = analysis.get("on_task", False)
        is_work_related = analysis.get("is_work_related", True)
        
        print(f"Activity check: on_task={is_on_task}, work_related={is_work_related}")
        
        # Send notification if needed - made more sensitive for testing
        if not is_on_task:  # Removed the is_work_related check to trigger more often
            print("Potential procrastination detected!")
            self.send_notification("You seem to be procrastinating. Time to get back to work!")
        else:
            print("User is on task, no notification needed")
        
        print(f"--- Monitoring cycle completed ---")

def main():
    monitor = ProcrastinationMonitor()
    
    # Schedule monitoring every X minutes
    schedule.every(SCREENSHOT_INTERVAL).minutes.do(monitor.monitor_cycle)
    
    # Schedule daily summary at 6 PM
    schedule.every().day.at("18:00").do(monitor.generate_daily_summary)
    
    print(f"Procrastination monitor started. Taking screenshots every {SCREENSHOT_INTERVAL} minutes.")
    
    # Run the first cycle immediately
    monitor.monitor_cycle()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main() 