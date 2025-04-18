from .chroma import ChromaDBHandler

# Create a single instance of ChromaDBHandler connected to Docker container
# Adjust host and port according to your Docker container configuration
chroma_client = ChromaDBHandler(host="localhost", port=8000)