import os
import time
import logging
import pyautogui
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from typing import Optional, List, Union

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TypeTextRequest(BaseModel):
    """Request schema for typing text"""
    text: str = Field(description="The text to type")
    interval: float = Field(default=0.01, description="Time interval between keypresses (in seconds)")

class PressKeysRequest(BaseModel):
    """Request schema for pressing specific keys"""
    keys: Union[str, List[str]] = Field(
        description="Key or list of keys to press (e.g., 'enter', 'ctrl+c', ['ctrl', 'alt', 'del'])"
    )
    presses: int = Field(default=1, description="Number of times to press the key(s)")
    interval: float = Field(default=0.1, description="Time interval between keypresses (in seconds)")

class HotKeyRequest(BaseModel):
    """Request schema for pressing hotkey combinations"""
    keys: List[str] = Field(
        description="List of keys to press simultaneously (e.g., ['ctrl', 'c'] for Ctrl+C)"
    )

def type_text(text: str, interval: float = 0.01) -> str:
    """
    Types the specified text with the keyboard at the current cursor position.
    
    Args:
        text: The text to type
        interval: Time interval between keypresses (in seconds)
        
    Returns:
        str: Confirmation message
    """
    try:
        # Use pyautogui to type the text
        pyautogui.write(text, interval=interval)
        logger.info(f"Typed text: {text[:20]}{'...' if len(text) > 20 else ''}")
        return f"Successfully typed text: {text[:20]}{'...' if len(text) > 20 else ''}"
    except Exception as e:
        logger.error(f"Error typing text: {str(e)}")
        return f"Error typing text: {str(e)}"

def press_keys(keys: Union[str, List[str]], presses: int = 1, interval: float = 0.1) -> str:
    """
    Presses the specified key(s) on the keyboard.
    
    Args:
        keys: Key or list of keys to press (e.g., 'enter', ['ctrl', 'c'])
        presses: Number of times to press the key(s)
        interval: Time interval between keypresses (in seconds)
        
    Returns:
        str: Confirmation message
    """
    try:
        # Handle both string and list formats
        if isinstance(keys, str):
            # Check if it's a compound hotkey like 'ctrl+c'
            if '+' in keys:
                key_parts = keys.split('+')
                with pyautogui.hold(key_parts[:-1]):
                    for _ in range(presses):
                        pyautogui.press(key_parts[-1])
                        time.sleep(interval)
                logger.info(f"Pressed hotkey: {keys} {presses} times")
                return f"Successfully pressed hotkey: {keys} {presses} times"
            else:
                # Single key press
                pyautogui.press(keys, presses=presses, interval=interval)
                logger.info(f"Pressed key: {keys} {presses} times")
                return f"Successfully pressed key: {keys} {presses} times"
        else:
            # List of keys (assume it's a hotkey combination if multiple keys)
            if len(keys) > 1:
                with pyautogui.hold(keys[:-1]):
                    for _ in range(presses):
                        pyautogui.press(keys[-1])
                        time.sleep(interval)
                logger.info(f"Pressed keys: {'+'.join(keys)} {presses} times")
                return f"Successfully pressed keys: {'+'.join(keys)} {presses} times"
            else:
                # Single key from a list
                pyautogui.press(keys[0], presses=presses, interval=interval)
                logger.info(f"Pressed key: {keys[0]} {presses} times")
                return f"Successfully pressed key: {keys[0]} {presses} times"
    except Exception as e:
        logger.error(f"Error pressing keys: {str(e)}")
        return f"Error pressing keys: {str(e)}"

def press_hotkey(keys: List[str]) -> str:
    """
    Presses a hotkey combination (multiple keys at once).
    
    Args:
        keys: List of keys to press simultaneously (e.g., ['ctrl', 'c'] for Ctrl+C)
        
    Returns:
        str: Confirmation message
    """
    try:
        # Use pyautogui's hotkey function
        pyautogui.hotkey(*keys)
        logger.info(f"Pressed hotkey: {'+'.join(keys)}")
        return f"Successfully pressed hotkey: {'+'.join(keys)}"
    except Exception as e:
        logger.error(f"Error pressing hotkey: {str(e)}")
        return f"Error pressing hotkey: {str(e)}"

# Create the LangChain structured tools
type_text_tool = StructuredTool(
    name="type_text",
    description="Type text at the current cursor position",
    func=type_text,
    args_schema=TypeTextRequest
)

press_keys_tool = StructuredTool(
    name="press_keys",
    description="Press specific key(s) on the keyboard",
    func=press_keys,
    args_schema=PressKeysRequest
)

hotkey_tool = StructuredTool(
    name="press_hotkey",
    description="Press a hotkey combination (multiple keys at once)",
    func=press_hotkey,
    args_schema=HotKeyRequest
)

# List of keyboard tools
keyboard_tools = [type_text_tool, press_keys_tool, hotkey_tool] 