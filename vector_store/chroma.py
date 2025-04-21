import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict
import uuid


class ChromaDBHandler:
    def __init__(self, host: str = "localhost", port: int = 8000):
        """Initialize ChromaDB client to connect to Docker container"""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False
            )
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        return self.model.encode(texts).tolist()

    def create_collection(self, collection_name: str) -> chromadb.Collection:
        try:
            return self.client.get_or_create_collection(name=collection_name)
        except ValueError:
            return self.client.get_collection(name=collection_name)

    def add_documents(self, collection_name: str, documents: List[str],
                      metadatas: Optional[List[Dict]] = None,
                      ids: Optional[List[str]] = None) -> None:
        """Add documents with embeddings"""
        collection = self.create_collection(collection_name)
        embeddings = self.embed(documents)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        if metadatas is None:
            metadatas = [{} for _ in documents]

        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            # metadatas=metadatas
        )

    def query_collection(self, collection_name: str, query_texts: List[str],
                         n_results: int = 5) -> Dict:
        collection = self.client.get_collection(name=collection_name)
        query_embeddings = self.embed(query_texts)

        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )
        return results

    def delete_collection(self, collection_name: str) -> None:
        self.client.delete_collection(collection_name)

    def delete_documents(self, collection_name: str, ids: List[str]) -> None:
        collection = self.client.get_collection(name=collection_name)
        collection.delete(ids=ids)

    def get_collection_items(self, collection_name: str) -> Dict:
        collection = self.client.get_collection(name=collection_name)
        return collection.get()

    def list_collections(self) -> List[str]:
        return [col.name for col in self.client.list_collections()]
