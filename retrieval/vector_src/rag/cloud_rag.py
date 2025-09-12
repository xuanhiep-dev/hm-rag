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


# Class Cloud RAG chain
class Cloud_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            <s>[INST] <<SYS>>\n You are a helpful assistant, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 
            Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that 
            your responses are socially unbiased and positive in nature.\
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not 
            correct. If you don't know the answer to a question, please response as language model you are not able to respone detailed to 
            these kind of question.\n<</SYS>>\n\n

            Ngữ cảnh:
            {context}

            Câu hỏi: {question}

            Trả lời:
            [/INST] 
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
