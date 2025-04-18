from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from vector_store import chroma_client

router = APIRouter()

class Question(BaseModel):
    question: str

@router.get("/")
async def root():
    return {"message": "Hello World"}

@router.post("/questions")
async def add_questions(questions: List[Question]):
    # Example of using the shared ChromaDB instance
    documents = [q.question for q in questions]
    chroma_client.add_documents("questions_collection", documents)
    return {"received_questions": questions}

@router.get("/questions")
async def get_questions():
    # Example of querying the shared ChromaDB instance
    try:
        results = chroma_client.get_collection_items("questions_collection")
        return {"questions": results}
    except ValueError:
        return {"questions": [], "message": "No questions collection found"}