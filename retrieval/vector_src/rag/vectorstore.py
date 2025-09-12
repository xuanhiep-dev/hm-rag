from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant as QdrantVectorStore


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
        self.embedding = embedding or HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Kết nối Qdrant Cloud / local
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

        # Tạo collection nếu chưa có
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size, distance=Distance.COSINE),
            )

    def upsert_documents(self, documents):
        """Nhúng và upsert list[Document] vào Qdrant"""
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding.embed_documents(texts)

        points = []
        for idx, (vec, doc) in enumerate(zip(embeddings, documents)):
            points.append(
                PointStruct(
                    id=idx,
                    vector=vec,
                    payload=doc.metadata | {"text": doc.page_content},
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def get_retriever(self, search_kwargs: dict = {"k": 5}):
        """Trả về retriever hợp chuẩn LangChain"""
        store = QdrantVectorStore.from_existing_collection(
            embedding=self.embedding,
            collection_name=self.collection_name,
            client=self.client,
        )
        return store.as_retriever(search_kwargs=search_kwargs)
