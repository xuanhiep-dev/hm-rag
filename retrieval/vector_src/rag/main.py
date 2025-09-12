from pydantic import BaseModel, Field
from src.rag.file_loader import Loader
from src.rag.vectorstore import VectorDB
from src.rag.cloud_rag import Cloud_RAG

from langchain_community.embeddings import HuggingFaceEmbeddings
QDRANT_URL = "https://b889a9b5-f641-48cd-bde4-b21a7baf2cd8.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.GSFP7G4NBbSY_6RB3qpY02hZKXRmsJDfjYYxnxn5jAY"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")


class InputQA(BaseModel):
    question: str = Field(..., title="Question to ask the model")


class OutputQA(BaseModel):
    answer: str = Field(..., title="Answer from the model")


def build_rag_chain(llm, data_dir, data_type):
    doc_loaded = Loader(file_type=data_type).load_dir(data_dir, workers=2)
    retriever = VectorDB(
        documents=doc_loaded,
        embedding=embedding_model,
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        collection_name="medical_collection"
    ).get_retriever(search_kwargs={"k": 5})
    rag_chain = Cloud_RAG(llm).get_chain(retriever)
    return rag_chain
