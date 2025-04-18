import chromadb
from chromadb.config import Settings
from typing import List, Optional, Dict
import os

class ChromaDBHandler:
    def __init__(self, host: str = "localhost", port: int = 8000):
        """Initialize ChromaDB client to connect to Docker container"""
        self.client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False
            )
        )

    def create_collection(self, collection_name: str) -> chromadb.Collection:
        """Create a new collection or get existing one"""
        try:
            return self.client.create_collection(name=collection_name)
        except ValueError:
            return self.client.get_collection(name=collection_name)

    def add_documents(self, collection_name: str, documents: List[str], 
                     metadatas: Optional[List[Dict]] = None, 
                     ids: Optional[List[str]] = None) -> None:
        """Add documents to a collection"""
        collection = self.create_collection(collection_name)
        if ids is None:
            ids = [str(i) for i in range(len(documents))]
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query_collection(self, collection_name: str, query_texts: List[str], 
                        n_results: int = 5) -> Dict:
        """Query documents from a collection"""
        collection = self.client.get_collection(collection_name)
        results = collection.query(
            query_texts=query_texts,
            n_results=n_results
        )
        return results

    def delete_collection(self, collection_name: str) -> None:
        """Delete an entire collection"""
        self.client.delete_collection(collection_name)

    def delete_documents(self, collection_name: str, ids: List[str]) -> None:
        """Delete specific documents from a collection"""
        collection = self.client.get_collection(collection_name)
        collection.delete(ids=ids)

    def get_collection_items(self, collection_name: str) -> Dict:
        """Get all items in a collection"""
        collection = self.client.get_collection(collection_name)
        return collection.get()

    def list_collections(self) -> List[str]:
        """List all available collections"""
        return [col.name for col in self.client.list_collections()]