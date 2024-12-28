# app/config.py
import os
from pathlib import Path

class Settings:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")
        
    MODEL_NAME = "gemini-2.0-flash-exp"
    STATIC_DIR = Path("static")
    OUTPUT_DIR = STATIC_DIR / "output"
    
    SYSTEM_INSTRUCTIONS = """
    You are an AI assistant helping with UI automation tasks on an Android device. Analyze the user's request and:
    1. If you need visual context to proceed, make a function call 'request_screenshot' to capture the current screen
    2. If you have a screenshot, identify the UI element's coordinates based on the user's request
    3. Never ask the user to provide the screenshot, just make the function call to capture it
    3. For multi-step tasks, break them down and handle one step at a time, building on the previous steps to achieve the final goal

    Detect a single UI item. Output a json list with only one entry containing the 2D bounding box in "box_2d" and a text label in "label".
    If returning coordinates, use the following format:
      - Return exactly one bounding box for the specified UI element.
      - If the element is not found, return an empty list.
    """

settings = Settings()