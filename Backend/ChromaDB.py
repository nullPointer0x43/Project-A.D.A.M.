from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from URLs import DEFAULT_CHROMA_HOST, DEFAULT_CHROMA_PORT, DEFAULT_COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL


class Chroma_client:
    def __init__(self, host=DEFAULT_CHROMA_HOST, port=DEFAULT_CHROMA_PORT):
        self.remote_client = chromadb.HttpClient(host=host, port=port)

        self.vector_db = Chroma(
            client=self.remote_client,
            collection_name=DEFAULT_COLLECTION_NAME,
            embedding_function=HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)
        )

    def add_to_db(self, text: str, metadata: dict, doc_id: str):
        self.vector_db.add_texts(
            texts=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        print(f"Successfully added document {doc_id} to ChromaDB.")

    def similarity_search(self, q):
        return self.vector_db.similarity_search(q)