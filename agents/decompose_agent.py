import re
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, create_react_agent


class DecomposeAgent:
    def __init__(self):
        # Load HuggingFace model (ví dụ Qwen2.5-3B từ HF)
        model_name = "Qwen/Qwen2.5-3B-Instruct"  # đổi thành model bạn muốn
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto"  # giảm RAM nếu cần
        ).to("cuda:0")

        # Tạo pipeline
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.35,
            device=0
        )

        # LangChain wrapper
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        self.llm = ChatHuggingFace(llm=llm)

        self.tools = [
            {
                "name": "Phân rã truy vấn",
                "description": "Phân rã câu truy vấn thành các câu hỏi con độc lập.",
                "func": self.decompose
            }
        ]

        # LangGraph agent
        self.agent = create_react_agent(self.llm, self.tools)

    def count_intents(self, query: str) -> int:
        """
        Xác định số lượng ý định (intents) trong câu truy vấn.
        Sử dụng LLM để phân tích số ý định có trong văn bản đầu vào.
        Args:
            query (str): câu truy vấn đầu vào.
        Returns:
            int: số ý định được phát hiện.
        """
        prompt = PromptTemplate.from_template(
            "Hãy tính số lượng ý định (intent) độc lập có trong câu truy vấn sau. "
            "Chỉ trả về một số nguyên duy nhất:\n{query}\nSố ý định: "
        )

        chain = prompt | self.llm
        for _ in range(3):
            response = chain.invoke({"query": query})
            match = re.search(r"\d+", response)
            if match:
                return int(match.group(0))
        return 1

    def decompose(self, query: str) -> List[str]:
        """
        Phân rã câu truy vấn. 
        Nếu số lượng ý định lớn hơn 1 thì tiến hành phân tách.
        Args:
            query (str): câu truy vấn đầu vào.
        Returns:
            List[str]: danh sách các câu hỏi con đã phân tách.
        """
        # intent_count = self.count_intents(query)
        # intent_count = min(intent_count, 3)
        intent_count = 3
        if intent_count > 1:
            return self._split_query(query, intent_count)
        return [query]

    def _split_query(self, query: str, intent_count: int) -> List[str]:
        """
        Thực hiện việc phân tách câu truy vấn thành các câu hỏi con.
        Args:
            query (str): câu truy vấn đầu vào.
            intent_count (int): số ý định cần phân rã.
        Returns:
            List[str]: danh sách câu hỏi con.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Bạn là một trợ lý AI hữu ích. "
             "Nhiệm vụ của bạn là phân tách truy vấn thành các câu hỏi con. "
             "Luôn trả lời bằng tiếng Việt, không sử dụng tiếng Anh."),
            ("user",
             "Hãy phân tách yêu cầu sau thành đúng {intent_count} câu hỏi con độc lập. "
             "Mỗi câu hỏi phải là một câu hoàn chỉnh bằng tiếng Việt. "
             "Không được viết lại theo nhiều cách khác nhau. "
             "Không giải thích. "
             "Chỉ xuất kết quả theo đúng định dạng sau:\n\n"
             "<câu hỏi con 1>\n"
             "<câu hỏi con 2>\n"
             "<câu hỏi con 3>\nKẾT THÚC\n\n"
             "Truy vấn: {query}\n\n"
             "Các câu hỏi con:")
        ])

        chain = prompt | self.llm
        response = chain.invoke({"query": query, "intent_count": intent_count})
        resp_text = response.content if hasattr(
            response, "content") else str(response)
        resp_text = resp_text.split("<|im_start|>assistant", 1)[-1]
        matches = [line.strip("-• \t")
                   for line in resp_text.splitlines() if line.strip()]

        return matches


# def run_decomposition(agent: DecomposeAgent, query: str) -> List[str]:
#     return agent.decompose(query)


# ---------------------- Run tests ----------------------
# if __name__ == "__main__":
#     agent = DecomposeAgent()
#     query = """Kiểm tra thời tiết hôm nay ở Thượng Hải, sau đó tóm tắt những tin khoa học mới nhất từ Đại học Fudan,
#         và cuối cùng so sánh ưu điểm và nhược điểm của Python và Java."""
#     subqueries = run_decomposition(agent, query)
#     print("Những yêu cầu cần thực hiện:")
#     for i, subq in enumerate(subqueries, 1):
#         print(f"{i}. {subq}")
