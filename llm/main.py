from pydantic import BaseModel
from fastapi import HTTPException
import ollama
import re

class ChatMessage(BaseModel):
    question: str
    context: str

class ChatResponse(BaseModel):
    response: str
    
class DeepSeekChat:
    def __init__(self):
        self.model_name = "deepseek-r1:7b"
    
    async def chat(self, message: ChatMessage) -> ChatResponse:
        try:
            prompt = f"""You are an AI assistant. Answer directly based on the provided context. Do not show your thinking process.Context:{message.context}Question:{message.question}"""
            
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                stream=False
            )
            final_answer = re.sub(r"<think>.*?</think>", "", response['message']['content'], flags=re.DOTALL).strip()
            
            return ChatResponse(
                response=final_answer
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error communicating with Ollama: {str(e)}"
            )