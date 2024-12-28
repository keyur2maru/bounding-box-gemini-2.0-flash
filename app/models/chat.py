# app/models/chat.py
import datetime
from typing import List, Optional, Dict
import uuid
from pydantic import BaseModel

class ChatMessage(BaseModel):
    text: str
    is_user: bool
    screenshot_path: Optional[str] = None

class ChatSession:
    def __init__(self):
        self.messages: List[ChatMessage] = []
        self.created_at: datetime = datetime.datetime.now()

class SessionManager:
    def __init__(self):
        self.active_sessions: Dict[str, ChatSession] = {}
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> tuple[str, ChatSession]:
        if session_id and session_id in self.active_sessions:
            return session_id, self.active_sessions[session_id]
        
        new_session_id = session_id or str(uuid.uuid4())
        self.active_sessions[new_session_id] = ChatSession()
        return new_session_id, self.active_sessions[new_session_id]

    def clear_session(self, session_id: str) -> bool:
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            return True
        return False