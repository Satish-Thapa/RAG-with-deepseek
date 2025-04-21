from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from vector_store import chroma_client
from llm.main import DeepSeekChat, ChatMessage, ChatResponse

router = APIRouter()
chat_client = DeepSeekChat()

class Question(BaseModel):
    question: str

class QueryRequest(BaseModel):
    query: str

@router.get("/")
async def root():
    return {"message": "Hello World"}

@router.post("/questions")
async def add_questions(questions: List[Question]):
    documents = [q.question for q in questions]
    chroma_client.add_documents("questions_collection", documents)
    return {"received_questions": questions}

@router.get("/collections")
async def get_collections():
    try:
        results = chroma_client.get_collection_items("questions_collection")
        return {"questions": results}
    except ValueError:
        return {"questions": [], "message": "No questions collection found"}

@router.post("/questions/search")
async def search_questions(query_request: QueryRequest):
    try:
        results = chroma_client.query_collection(
            "questions_collection",
            query_texts=[query_request.query],
            n_results=5
        )
        
        all_documents = [doc for docs in results["documents"] for doc in docs]
        context = "\n".join(all_documents)
        
        chat_response = await chat_client.chat(ChatMessage(
            question=query_request.query,
            context=context
        ))
        
        return {
            "results": results,
            "ai_response": chat_response.response
        }
    except ValueError:
        return {"results": [], "message": "No matching questions found or collection does not exist"}

@router.post("/chat", response_model=ChatResponse)
async def chat_with_deepseek(messages: List[ChatMessage]):
    return await chat_client.chat(messages)