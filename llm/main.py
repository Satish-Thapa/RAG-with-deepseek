from typing import List
from pydantic import BaseModel
from decouple import config
from deepseek import DeepSeekAPI
from fastapi import HTTPException

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatResponse(BaseModel):
    response: str
    
class DeepSeekChat:
    def __init__(self):
        api_key = config('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
        self.client = DeepSeekAPI(api_key=api_key)
    
    async def chat(self, messages: List[ChatMessage], model: str = "deepseek-coder-33b-base") -> ChatResponse:
        try:
            chat_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            
            response = self.client.chat_completion(
                api_key=self.api_key,
                messages=chat_messages,
                model=model,
                temperature=0.7,
                max_tokens=1000
            )
            
            return ChatResponse(
                response=response["choices"][0]["message"]["content"]
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error communicating with DeepSeek API: {str(e)}"
            )