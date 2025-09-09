from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

from retrieval.base_retrieval import BaseRetrieval

import os
os.environ["http_proxy"] = "http://127.0.0.1:11434"
os.environ["https_proxy"] = "http://127.0.0.1:11434"

class VectorRetrieval(BaseRetrieval):
    def __init__(self, config):
        self.config = config
        
        self.client = LightRAG(
        working_dir=self.config.working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name=self.config.llm_model_name,
        llm_model_max_async=160,
        llm_model_max_token_size=65536,
        llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 65536}},
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model="nomic-embed-text", host="http://localhost:11434"
            ),
        ),
        )
        self.results = []

    
    def find_top_k(self, query):
        # self.results = self.client.query(query, 
                                        #  param=QueryParam(mode=self.config.mode, 
                                        #                   top_k=self.config.top_k),
                                        #  param=QueryParam(mode="naive"))
        prompt = "Context: N/A\nQuestion: Which figure of speech is used in this text?\nSing, O goddess, the anger of Achilles son of Peleus, that brought countless ills upon the Achaeans.\n—Homer, The Iliad\nOptions: (A) chiasmus (B) apostrophe\nAnswer:\nSummary the output with format 'Answer: The answer is A, B, C, D, E or FAILED. \n BECAUSE: '"
        self.results = self.client.query(prompt, 
                                        #  param=QueryParam(mode=self.config.mode))
                                                        #  , top_k=self.config.top_k))
                                         param=QueryParam(mode="naive"))
        return self.results
    
    