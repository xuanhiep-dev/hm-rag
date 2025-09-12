from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient


class VectorDB:
    def __init__(
        self,
        documents=None,
        embedding=None,
        collection_name: str = "my_collection",
        qdrant_url: str = None,
        qdrant_api_key: str = None,
    ) -> None:
        self.embedding = embedding or HuggingFaceEmbeddings()
        self.collection_name = collection_name

        # Kết nối tới Qdrant Cloud
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

        # Nếu có documents thì nạp vào Qdrant
        if documents:
            self.db = Qdrant.from_documents(
                documents=documents,
                embedding=self.embedding,
                collection_name=self.collection_name,
                client=self.client
            )
        else:
            # Nếu không có docs thì chỉ khởi tạo kết nối
            self.db = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=self.embedding
            )

    def get_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: dict = {"k": 10}
    ):
        retriever = self.db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        return retriever
