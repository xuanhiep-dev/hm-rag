import re
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate


class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, text: str) -> str:
        return self.extract_answer(text)

    def extract_answer(self, text_response: str, pattern: str = r"Answer:\s*(.*)") -> str:
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text
        else:
            return text_response


# Class Offline RAG chain
class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Bạn là một trợ lý hữu ích cho các nhiệm vụ hỏi đáp y khoa. 
            Hãy sử dụng phần ngữ cảnh dưới đây để trả lời câu hỏi một cách chính xác. 
            Câu trả lời cần ngắn gọn nhưng đầy đủ ý (2–3 câu), bao quát các điểm chính mà không lặp lại. 
            Nếu trong ngữ cảnh không có thông tin, hãy trả lời: "Tôi không biết."

            Ngữ cảnh:
            {context}

            Câu hỏi: {question}

            Trả lời:
            """
        )
        self.str_parser = Str_OutputParser()

    def get_chain(self, retriever):
        input_data = {
            "context": retriever | self.format_docs,
            "question": RunnablePassthrough()
        }

        rag_chain = (
            input_data
            | self.prompt
            | self.llm
            | self.str_parser
        )
        return rag_chain

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
