from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from retrieval.base_retrieval import BaseRetrieval
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch


class VectorRetrieval(BaseRetrieval):
    def __init__(self, config):
        self.config = config

        # ─────────────────────────────
        # 🧠 Dùng model Hugging Face thay Ollama
        # ─────────────────────────────
        embed_model_name = getattr(
            config, "embed_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        llm_model_name = getattr(
            config, "llm_model_name", "Qwen/Qwen2.5-0.5B-Instruct")

        # Embedding model (text -> vector)
        from sentence_transformers import SentenceTransformer
        self.embed_model = SentenceTransformer(embed_model_name)

        # LLM model (nếu cần xử lý query hoặc reasoning)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # ─────────────────────────────
        # 🔧 Khởi tạo LightRAG với embedding Hugging Face
        # ─────────────────────────────
        self.client = LightRAG(
            working_dir=self.config.working_dir,
            llm_model_func=self._local_llm_complete,  # custom HF completion
            llm_model_name=llm_model_name,
            llm_model_max_async=64,
            embedding_func=EmbeddingFunc(
                embedding_dim=self.embed_model.get_sentence_embedding_dimension(),
                max_token_size=8192,
                func=lambda texts: self.embed_model.encode(
                    texts, convert_to_numpy=True).tolist(),
            ),
        )
        self.results = []

    # ─────────────────────────────
    # 🧩 Tạo hàm thay thế cho ollama_model_complete
    # ─────────────────────────────
    def _local_llm_complete(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            self.llm_model.device)
        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text

    # ─────────────────────────────
    # 🔍 Tìm kiếm vector + reasoning
    # ─────────────────────────────
    def find_top_k(self, query):
        # Nếu cần custom prompt: có thể thêm phần Context / Question ở đây
        param = QueryParam(mode=getattr(self.config, "mode",
                           "naive"), top_k=self.config.top_k)
        self.results = self.client.query(query, param=param)
        return self.results
