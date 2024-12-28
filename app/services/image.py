from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import json
import asyncio
from pathlib import Path
from app.config import settings

class ImageService:
    @staticmethod
    async def process_uploaded_image(contents: bytes) -> tuple[Image.Image, Image.Image]:
        original_img = Image.open(BytesIO(contents))
        resized_img = original_img.resize((1024, 1024), Image.Resampling.LANCZOS)
        return original_img, resized_img
    
    @staticmethod
    async def save_processed_images(original_img: Image.Image, resized_img: Image.Image, 
                                  response_text: str) -> tuple[Path, Path]:
        output_dir = settings.OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().timestamp()
        resized_filename = output_dir / f"annotated_resized_{timestamp}.png"
        original_filename = output_dir / f"annotated_original_{timestamp}.png"
        
        resized_result = await ImageService.plot_bounding_boxes(
            resized_img.copy(), response_text, is_resized=True
        )
        original_result = await ImageService.plot_bounding_boxes(
            original_img.copy(), response_text, is_resized=False,
            original_dims=original_img.size
        )
        
        await ImageService.save_image(resized_result, resized_filename)
        await ImageService.save_image(original_result, original_filename)
        
        return resized_filename, original_filename

    @staticmethod
    async def plot_bounding_boxes(image: Image.Image, bounding_boxes: str, 
                                is_resized: bool = True, original_dims: tuple = None) -> Image.Image:
        width, height = image.size
        draw = ImageDraw.Draw(image)
        color = 'red'
        
        try:
            json_boxes = await ImageService.parse_json(bounding_boxes)
            boxes = json.loads(json_boxes)
            
            for box in boxes:
                coords = None
                if "box_2d" in box:
                    coords = ImageService.calculate_absolute_coordinates(
                        box["box_2d"], width, height, is_normalized=True
                    )
                elif "bounding_box" in box:
                    coords = ImageService.calculate_absolute_coordinates(
                        box["bounding_box"], width, height, is_normalized=True
                    )
                elif all(key in box for key in ["x", "y", "width", "height"]):
                    # First normalize the coordinates (assuming they're from 1024x1024)
                    x1 = box["x"] / 1024  # Normalize to 0-1
                    y1 = box["y"] / 1024
                    x2 = (box["x"] + box["width"]) / 1024
                    y2 = (box["y"] + box["height"]) / 1024
                    
                    # Now convert to absolute coordinates for this image
                    coords = (
                        int(x1 * width),
                        int(y1 * height),
                        int(x2 * width),
                        int(y2 * height)
                    )
                else:
                    print(f"Unknown bounding box format: {box}")
                    continue
                
                if coords:
                    ImageService.draw_box_with_label(draw, coords, box.get("label"), color)
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {bounding_boxes}")
        
        return image

    @staticmethod
    def calculate_absolute_coordinates(coords, width, height, is_normalized=True):
        """
        Convert coordinates to absolute pixel coordinates.
        
        Args:
            coords: List of coordinates
            width: Image width
            height: Image height
            is_normalized: If True, coords are 0-1000 range. If False, they're absolute pixels.
        """
        if is_normalized:
            # Convert from 0-1000 range to absolute pixels
            abs_y1 = int(coords[0]/1000 * height)
            abs_x1 = int(coords[1]/1000 * width)
            abs_y2 = int(coords[2]/1000 * height)
            abs_x2 = int(coords[3]/1000 * width)
        else:
            # Already in pixels, just need to scale if image dimensions differ
            abs_x1, abs_y1, abs_x2, abs_y2 = coords
        
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
            
        return abs_x1, abs_y1, abs_x2, abs_y2

    @staticmethod
    def draw_box_with_label(draw, coords, label, color):
        draw.rectangle(((coords[0], coords[1]), (coords[2], coords[3])), 
                      outline=color, width=4)
        
        if label:
            font = ImageFont.load_default()
            draw.text((coords[0] + 8, coords[1] - 20), label, fill=color, font=font)

    @staticmethod
    async def save_image(image: Image.Image, filename: Path):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, image.save, filename)

    @staticmethod
    async def parse_json(json_output: str) -> str:
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                json_output = "\n".join(lines[i+1:])
                json_output = json_output.split("```")[0]
                break
        return json_output