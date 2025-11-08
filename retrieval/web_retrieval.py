from langchain_community.utilities import GoogleSerperAPIWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from retrieval.base_retrieval import BaseRetrieval
import torch
import traceback


class WebRetrieval(BaseRetrieval):
    def __init__(self, config):
        self.config = config
        self.search_engine = "Google"

        # ─────────────────────────────
        # 🔍 Search Engine
        # ─────────────────────────────
        self.client = GoogleSerperAPIWrapper(
            serper_api_key=self.config.serper_api_key,
            gl="cn",
            hl="en",
            num=self.config.top_k
        )

        # ─────────────────────────────
        # 🧠 Language Model (HuggingFace)
        # ─────────────────────────────
        model_name = getattr(config, "llm_model_name",
                             "Qwen/Qwen2.5-0.5B-Instruct")
        print(f"[WebRetrieval] Using model: {model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        except Exception as e:
            print(
                "⚠️ Không thể tải model lớn, fallback sang model nhỏ 'Qwen/Qwen2.5-0.5B-Instruct'")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-0.5B-Instruct")
            self.model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-0.5B-Instruct",
                torch_dtype=torch.float16,
                device_map="auto"
            )

        # Tạo pipeline sinh văn bản
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            temperature=0.35,
        )

        self.results = []

    # ─────────────────────────────
    # 🔎 Format kết quả tìm kiếm
    # ─────────────────────────────
    def format_results(self, results):
        max_results = 3
        processed = []
        if 'organic' in results:
            for item in results['organic'][:max_results]:
                processed.append(
                    f"[{item.get('title', 'No title')}]\n"
                    f"{item.get('snippet', 'No snippet')}\n"
                    f"Link: {item.get('link')}\n"
                )

        if 'answerBox' in results:
            answer = results['answerBox']
            processed.insert(
                0, f"Direct answer: {answer.get('answer', '')}\nSource: {answer.get('link', '')}\n"
            )

        return "\n".join(processed) or "No relevant results found."

    # ─────────────────────────────
    # 🧩 Sinh tóm tắt bằng HF pipeline
    # ─────────────────────────────
    def generation(self, results):
        try:
            prompt = (
                "Tóm tắt ngắn gọn nội dung dưới đây bằng tiếng Việt:\n"
                f"{results}\n\n"
                "→ Trả lời súc tích, có liên quan trực tiếp đến câu hỏi."
            )
            out = self.pipe(prompt)[0]["generated_text"]
            return out
        except Exception as e:
            print("⚠️ Lỗi trong self.generation:", e)
            traceback.print_exc()
            return "Không thể sinh câu trả lời từ web."

    # ─────────────────────────────
    # 🔍 Tìm kiếm top-k và sinh câu trả lời
    # ─────────────────────────────
    def find_top_k(self, query):
        try:
            search_results = self.client.results(query)
            formatted = self.format_results(search_results)

            # Chuẩn hóa kiểu dữ liệu
            query_text = query if isinstance(
                query, str) else "\n".join(map(str, query))
            joined_text = formatted + "\n\nCâu hỏi: " + query_text

            summary = self.generation(joined_text)
            self.results = summary
            return summary
        except Exception as e:
            print("⚠️ Lỗi trong find_top_k:", e)
            traceback.print_exc()
            return "Không thể tìm kiếm hoặc tóm tắt nội dung web."
