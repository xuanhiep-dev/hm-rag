from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorDB:
    def __init__(
        self,
        documents=None,
        embedding=HuggingFaceEmbeddings()
    ) -> None:
        self.embedding = embedding
        self.db = self._build_db(documents)

    def _build_db(self, documents):
        db = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding
        )
        return db

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
