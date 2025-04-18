from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

class Question(BaseModel):
    question: str

@router.get("/")
async def root():
    return {"message": "Hello World"}

@router.post("/questions")
async def add_questions(questions: List[Question]):
    return {"received_questions": questions}