from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from retrieval.base_retrieval import BaseRetrieval
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch


class GraphRetrieval(BaseRetrieval):
    def __init__(self, config):
        self.config = config

        # ─────────────────────────────
        # 🧠 Dùng Hugging Face thay Ollama
        # ─────────────────────────────
        llm_model_name = getattr(
            config, "llm_model_name", "Qwen/Qwen2.5-0.5B-Instruct")
        embed_model_name = getattr(
            config, "embed_model_name", "sentence-transformers/all-MiniLM-L6-v2")

        # Load embedding model (text → vector)
        self.embed_model = SentenceTransformer(embed_model_name)

        # Load ngôn ngữ model (dùng cho reasoning)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # ─────────────────────────────
        # ⚙️ Khởi tạo LightRAG (dùng local model)
        # ─────────────────────────────
        self.client = LightRAG(
            working_dir=self.config.working_dir,
            llm_model_func=self._local_llm_complete,
            llm_model_name=llm_model_name,
            llm_model_max_async=64,
            embedding_func=EmbeddingFunc(
                embedding_dim=self.embed_model.get_sentence_embedding_dimension(),
                max_token_size=8192,
                func=lambda texts: self.embed_model.encode(
                    texts, convert_to_numpy=True
                ).tolist(),
            ),
        )
        self.results = []

    # ─────────────────────────────
    # 🧩 Hàm thay thế ollama_model_complete
    # ─────────────────────────────
    def _local_llm_complete(self, prompt: str) -> str:
        """Sinh văn bản bằng model Hugging Face thay vì Ollama."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            self.llm_model.device)
        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.strip()

    # ─────────────────────────────
    # 🔍 Truy vấn đồ thị
    # ─────────────────────────────
    def find_top_k(self, query):
        try:
            param = QueryParam(
                mode=getattr(self.config, "mode", "mix"),
                top_k=getattr(self.config, "top_k", 3)
            )
            self.results = self.client.query(query, param=param)

            # 🔧 Normalize để downstream luôn nhận str
            if self.results is None:
                self.results = ""
            elif isinstance(self.results, list):
                self.results = "\n".join(map(str, self.results))
            else:
                self.results = str(self.results)

            return self.results.strip()
        except Exception as e:
            print(f"⚠️ Lỗi trong GraphRetrieval.find_top_k: {e}")
            return ""
