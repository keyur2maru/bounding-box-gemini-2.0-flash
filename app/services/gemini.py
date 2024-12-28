# app/services/gemini.py
import logging
from google import genai
from google.genai import types
from app.config import settings
import asyncio

logger = logging.getLogger(__name__)
request_screenshot_schema = {'name': 'request_screenshot'}

class GeminiService:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        
    async def generate_content(self, messages, prompt, image=None):
        contents = [*[msg.text for msg in messages], prompt]
        if image:
            contents.append(image)
        
        logger.info("Generating content with Gemini: %s", contents)

            
        return await asyncio.to_thread(
            self.client.models.generate_content,
            model=settings.MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=settings.SYSTEM_INSTRUCTIONS,
                response_modalities=["TEXT"],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=False,
                ),
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="AUTO",
                    )
                ),
                tools = [
                    {'function_declarations': [request_screenshot_schema]}
                ]
            )
        )