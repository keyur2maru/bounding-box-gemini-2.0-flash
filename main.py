# main.py
import os
import logging
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types
import json
from io import BytesIO
import aiofiles
import asyncio
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Gemini client
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

client = genai.Client(api_key=api_key)

# Model configuration
MODEL_NAME = "gemini-2.0-flash-exp"
BOUNDING_BOX_SYSTEM_INSTRUCTIONS = """
Return bounding boxes as a JSON array with labels. Never return masks or code fencing. 
Return exactly one bounding box for the specified UI element.
"""
SAFETY_SETTINGS = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]

async def parse_json(json_output: str) -> str:
    """Parse JSON output from Gemini response."""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output

async def plot_bounding_boxes(image: Image.Image, bounding_boxes: str, is_resized: bool = True, original_dims: tuple = None) -> Image.Image:
    """Plot bounding boxes on the image with labels."""
    width, height = image.size
    draw = ImageDraw.Draw(image)
    color = 'red'
    
    bounding_boxes = await parse_json(bounding_boxes)
    
    try:
        for bounding_box in json.loads(bounding_boxes):
            coords = bounding_box["box_2d"]
            
            if is_resized:
                abs_y1 = int(coords[0]/1000 * 1024)
                abs_x1 = int(coords[1]/1000 * 1024)
                abs_y2 = int(coords[2]/1000 * 1024)
                abs_x2 = int(coords[3]/1000 * 1024)
            else:
                orig_width, orig_height = original_dims
                abs_y1 = int(coords[0]/1000 * orig_height)
                abs_x1 = int(coords[1]/1000 * orig_width)
                abs_y2 = int(coords[2]/1000 * orig_height)
                abs_x2 = int(coords[3]/1000 * orig_width)
            
            if abs_x1 > abs_x2:
                abs_x1, abs_x2 = abs_x2, abs_x1
            if abs_y1 > abs_y2:
                abs_y1, abs_y2 = abs_y2, abs_y1
            
            draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)
            
            if "label" in bounding_box:
                font = ImageFont.load_default()
                draw.text((abs_x1 + 8, abs_y1 - 20), bounding_box["label"], fill=color, font=font)
                
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {bounding_boxes}")
    
    return image

async def save_image(image: Image.Image, filename: str):
    """Save image asynchronously."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, image.save, filename)

@app.get("/")
async def read_root():
    """Serve the index.html file."""
    return FileResponse("static/index.html")

@app.post("/analyze/")
async def analyze_image(
    file: UploadFile = File(...),
    prompt: str = Form(None)
):
    try:
        logger.info(f"Received file: {file.filename}")
        logger.info(f"Content type: {file.content_type}")
        if prompt:
            logger.info(f"Received prompt: '{prompt}'")
        else:
            prompt = "Detect the 2d bounding box of the 'General' button in this iOS settings screenshot"
            logger.info(f"Using default prompt: '{prompt}'")
        
        # Read the file once
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")
        
        original_img = Image.open(BytesIO(contents))
        original_width, original_height = original_img.size
        
        # Create 1024x1024 version
        target_size = (1024, 1024)
        resized_img = original_img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Use the provided prompt or default value
        
        # Generate content with Gemini
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL_NAME,
            contents=[prompt, resized_img],
            config=types.GenerateContentConfig(
                system_instruction=BOUNDING_BOX_SYSTEM_INSTRUCTIONS,
                temperature=0.5,
                safety_settings=SAFETY_SETTINGS,
            )
        )
        
        # Create output directory if it doesn't exist
        output_dir = Path("static/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process both versions
        resized_annotated = resized_img.copy()
        original_annotated = original_img.copy()
        
        resized_result = await plot_bounding_boxes(resized_annotated, response.text, is_resized=True)
        original_result = await plot_bounding_boxes(
            original_annotated, 
            response.text,
            is_resized=False,
            original_dims=(original_width, original_height)
        )
        
        # Save images
        resized_filename = output_dir / "annotated_resized.png"
        original_filename = output_dir / "annotated_original.png"
        
        await save_image(resized_result, resized_filename)
        await save_image(original_result, original_filename)
        
        return JSONResponse({
            "message": "Analysis complete",
            "resized_image": "/static/output/annotated_resized.png",
            "original_image": "/static/output/annotated_original.png",
            "gemini_response": response.text
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)