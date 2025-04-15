from langchain_community.vectorstores import Chroma
import ollama
import gradio as gr
from langchain_community.embeddings import OllamaEmbeddings
import re

# Load the embedding function
embedding_function = OllamaEmbeddings(model="deepseek-r1:7b")

def query_chromadb(user_question):

    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    retriever = vectorstore.as_retriever()

    # Retrieve relevant chunks from the database
    retrieved_docs = retriever.invoke(user_question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Query the LLM with the retrieved context
    formatted_prompt = (
        "You are an assistant for question-answering tasks.\n"
        "Use the following pieces of retrieved context to answer the question.\n"
        "If no context is present or if you don't know the answer, just say that you don't know.\n"
        "Do not make up the answer unless it is there in the provided context.\n"
        "Keep the answer concise and to the point with regard to the question.\n"
        f"Question:\n{user_question}\n\nContext:\n{context}\nAnswer:"
    )
    response = ollama.chat(
        model="deepseek-r1:7b",
        messages=[{"role": "user", "content": formatted_prompt}],
    )

    response_content = response["message"]["content"]
    final_answer = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()
    return final_answer

# Gradio interface
def gradio_interface(user_question):
    return query_chromadb(user_question)

interface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Ask a question about test papers"),
    outputs="text",
    title="Test Paper Q&A",
    description="Ask questions about test papers stored in the ChromaDB.",
)

if __name__ == "__main__":
    interface.launch()