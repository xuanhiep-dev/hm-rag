from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer
from langchain_qdrant import QdrantVectorStore


class VectorDB:
    def __init__(
        self,
        collection_name: str = "my_collection",
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        embedding=None,
        vector_size: int = 384,   # tuỳ model embedding, ví dụ all-MiniLM-L6-v2 = 384
    ) -> None:
        self.collection_name = collection_name
        self.embedding = embedding or SentenceTransformer(
            model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
        )

        # Kết nối Qdrant Cloud / local
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

    def get_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: dict = {"k": 10}
    ):
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding,
        )
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        return retriever
