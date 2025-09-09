
from langchain_community.utilities import GoogleSerperAPIWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import Ollama

from retrieval.base_retrieval import BaseRetrieval

class WebRetrieval(BaseRetrieval):
    def __init__(self, config):
        self.config = config
        self.search_engine = "Google"  # Default search engine
        self.client = GoogleSerperAPIWrapper(
            serper_api_key=self.config.serper_api_key,
            gl="cn",
            hhl="en",
            num=self.config.top_k
        )
        self.generator = self.config.web_llm_model_name

        self.llm = Ollama(
            model="qwen2.5:7b", 
            temperature=0.35,
        )        
        self.results = []

    def format_results(self, results):
        """Format the search results"""
        max_results = 1
        processed = []
        if 'organic' in results:
            for item in results['organic'][:max_results]:
                processed.append(f"[{item.get('title', 'No title')}]\n{item.get('snippet', 'No snippet')}\nLink:{item.get('link')}\n")
        
        if 'answerBox' in results:
            answer = results['answerBox']
            processed.insert(0, f"Direct answer：{answer.get('answer', '')}\nSource:{answer.get('link', '')}\n")
        
        return "\n".join(processed) or "No relevant results found"
    
    def generation(self, results):
        # 使用 Ollama 模型生成回答
        answer = self.llm(results)
        return answer
        
    
    def find_top_k(self, query):
        self.results = self.client.results(query)
        self.results = self.format_results(self.results)
        self.results = self.generation(self.results + "\n" + query)
        return self.results
    
    