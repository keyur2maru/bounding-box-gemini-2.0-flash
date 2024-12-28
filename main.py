# app/main.py
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.routes import router
from app.config import settings

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)