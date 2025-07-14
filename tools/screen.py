import os
import logging
import tempfile
import pyautogui
from PIL import Image
import google.generativeai as genai
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from typing import Optional, List
import time
import atexit
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the Gemini model
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not found in environment variables")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# Keep track of temporary files for cleanup
temp_files_to_cleanup = []

# Function to clean up temp files at exit
def cleanup_temp_files():
    global temp_files_to_cleanup
    for file_path in temp_files_to_cleanup:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to clean up temp file {file_path}: {e}")

# Register cleanup function to run at exit
atexit.register(cleanup_temp_files)

class ScreenCaptureRequest(BaseModel):
    """Request schema for screen capture and analysis"""
    query: str = Field(description="Question or description of what to look for on the screen")
    region: Optional[List[int]] = Field(
        default=None, 
        description="Optional region to capture [x, y, width, height]. If not provided, captures entire screen."
    )

def analyze_screen(query: str, region: Optional[List[int]] = None) -> str:
    """
    Captures the screen or a region of it and analyzes it using Gemini Vision model.
    
    Args:
        query: Question or instruction about what to look for in the image
        region: Optional [x, y, width, height] to capture a specific portion of screen
        
    Returns:
        str: Description or analysis of what's on screen
    """
    screenshot_path = None
    try:
        # Create a temporary file for the screenshot
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            screenshot_path = temp_file.name
            # Add to global list for cleanup on exit
            temp_files_to_cleanup.append(screenshot_path)
            
        # Capture screenshot (either full screen or region)
        if region:
            x, y, width, height = region
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
        else:
            screenshot = pyautogui.screenshot()
        
        # Save screenshot to temp file
        screenshot.save(screenshot_path)
        logger.info(f"Screenshot captured to {screenshot_path}")
        
        # Make sure the file is closed
        screenshot.close()
        
        # Small delay to ensure file is released by PIL
        time.sleep(0.1)
        
        # Load the image for processing
        with Image.open(screenshot_path) as image:
            # Initialize Gemini model
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Create prompt that instructs the model to analyze the image
            prompt = f"""
            Analyze this screenshot from a user's computer screen. 
            
            User query: {query}
            
            Provide a detailed response about what you see in relation to the query.
            Include:
            - Text content visible on screen
            - UI elements and their layout
            - Any information relevant to answering the user's query
            
            Be specific and detailed in your analysis.
            """
            
            # Generate content with the image
            response = model.generate_content([prompt, image])
        
        # Schedule file deletion instead of immediate deletion
        def delayed_delete():
            try:
                time.sleep(1)  # Wait a second before trying to delete
                if os.path.exists(screenshot_path):
                    os.remove(screenshot_path)
                    # Remove from cleanup list if successfully deleted
                    if screenshot_path in temp_files_to_cleanup:
                        temp_files_to_cleanup.remove(screenshot_path)
                    logger.info(f"Deleted temp file: {screenshot_path}")
            except Exception as e:
                logger.warning(f"Could not delete temp file {screenshot_path}: {e}")
                # File will be cleaned up on exit
        
        # Start deletion in background thread
        threading.Thread(target=delayed_delete, daemon=True).start()
        
        # Return the analysis
        return response.text
        
    except Exception as e:
        logger.error(f"Error in screen analysis: {str(e)}")
        return f"Error analyzing screen: {str(e)}"

# Create the LangChain structured tool
screen_analysis_tool = StructuredTool(
    name="analyze_screen",
    description="Capture and analyze what's currently visible on the user's screen",
    func=analyze_screen,
    args_schema=ScreenCaptureRequest
)

# List of screen tools
screen_tools = [screen_analysis_tool]
