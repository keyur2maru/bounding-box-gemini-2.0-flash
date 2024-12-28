# app/api/routes.py
import datetime
import uuid
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import logging
from app.services.gemini import GeminiService
from app.services.image import ImageService
from app.models.chat import ChatMessage, SessionManager
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

session_manager = SessionManager()
gemini_service = GeminiService()
image_service = ImageService()

@router.post("/process_prompt/")
async def process_prompt(
    prompt: str = Form(...),
    file: UploadFile = Form(None),
    session_id: Optional[str] = Form(None)
):
    session_id, session = session_manager.get_or_create_session(session_id)
    try:
        # Handle initial prompt without image
        if not file:
            session.messages.append(ChatMessage(
                text=prompt,
                is_user=True
            ))

            # Generate initial response
            response = await gemini_service.generate_content(
                session.messages, prompt
            )

            logger.debug("Gemini response: %s", response)

            actions = []
            messages = []

            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if part.text:
                        messages.append(part.text.strip())
                    elif part.function_call:
                        logger.debug(f"Function call detected: {part.function_call}")
                        actions.append({
                            "function_name": part.function_call.name,
                            "args": part.function_call.args,
                        })

            # Handle screenshot request
            if any(action["function_name"] == "request_screenshot" for action in actions):
                session.messages.append(ChatMessage(
                    text="Screenshot requested by model.",
                    is_user=False
                ))
                logger.debug("Chat history: %s", session.messages)
                return JSONResponse({
                    "action": "screenshot",
                    "message": "Screenshot requested",
                    "session_id": session_id
                })

            # Return text response if no screenshot needed
            if messages:
                response_text = " ".join(messages)
                session.messages.append(ChatMessage(
                    text=response_text,
                    is_user=False
                ))
                logger.debug("Chat history: %s", session.messages)
                return JSONResponse({
                    "action": "success",
                    "gemini_response": response_text,
                    "session_id": session_id
                })

        # Handle prompt with image
        else:
            contents = await file.read()
            original_img, resized_img = await image_service.process_uploaded_image(contents)
            
            # Save screenshot for session history
            screenshot_path = settings.OUTPUT_DIR / f"screenshot_{datetime.datetime.now()}_{session_id}.png"
            await image_service.save_image(original_img, screenshot_path)

            # Add screenshot message to session
            session.messages.append(ChatMessage(
                text="Screenshot provided",
                is_user=True,
                screenshot_path=str(screenshot_path)
            ))

            logger.debug("Chat history: %s", session.messages)

            # Generate content with image
            response = await gemini_service.generate_content(
                session.messages, prompt, resized_img
            )

            logger.debug("Gemini response with image: %s", response)

            # Process and save annotated images
            resized_filename, original_filename = await image_service.save_processed_images(
                original_img, resized_img, response.text
            )

            # Add response to session
            session.messages.append(ChatMessage(
                text=response.text,
                is_user=False
            ))

            logger.debug("Chat history: %s", session.messages)

            return JSONResponse({
                "action": "image_analysis",
                "message": "Analysis complete",
                "resized_image": f"/static/output/{resized_filename.name}",
                "original_image": f"/static/output/{original_filename.name}",
                "gemini_response": response.text,
                "session_id": session_id
            })

        # Handle case with no valid parts
        logger.warning("No valid text or actionable parts in Gemini response.")
        return JSONResponse({
            "action": "error",
            "message": "No valid response parts found.",
            "session_id": session_id
        })

    except Exception as e:
        logger.error(f"Error processing prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear_session/")
async def clear_session(session_id: Optional[str] = Form(None)):
    if session_id:
        success = session_manager.clear_session(session_id)
        if success:
            return JSONResponse({"message": "Session cleared", "session_id": session_id})
    return JSONResponse({"message": "No session to clear", "session_id": session_id})