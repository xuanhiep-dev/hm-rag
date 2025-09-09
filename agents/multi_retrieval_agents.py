from retrieval.vector_retrieval import VectorRetrieval
from retrieval.graph_retrieval import GraphRetrieval
from retrieval.web_retrieval import WebRetrieval
from summary_agent import SummaryAgent
from decompose_agent import DecomposeAgent

from typing import List
import json

class MRetrievalAgent():
    def __init__(self, config):
        self.config = config
        self.vector_retrieval = VectorRetrieval(config)
        self.graph_retrieval = GraphRetrieval(config)
        self.web_retrieval = WebRetrieval(config)
        self.sum_agent = SummaryAgent(config)
        self.dec_agent = DecomposeAgent(config)
        
    def predict(self, problems, shot_qids, qid):
        problem = problems[qid]
        question = problem['question']
        question = self.dec_agent.decompose(question)
        
        vector_response = self.vector_retrieval.find_top_k(question)
        graph_response = self.graph_retrieval.find_top_k(question)
        web_response = self.web_retrieval.find_top_k(question)
        

        all_messages = ["Vector Retrieval Agent:\n" + vector_response + "\n", 
                        "Graph Retrieval Agent:\n" + graph_response + "\n"
                        "Graph Retrieval Agent:\n" + graph_response + "\n"]
        
        final_ans, final_messages = self.sum(problems, shot_qids, qid, all_messages)
        return final_ans, final_messages
        
        
    def sum(self, problems, shot_qids, qid, sum_question):
        final_ans, all_messages = self.sum_agent.summarize(problems, shot_qids, qid, sum_question)
        # def extract_final_answer(agent_response):
        #     try:
        #         response_dict = json.loads(agent_response)
        #         answer = response_dict.get("Answer", None)
        #         return answer
        #     except:
        #         return agent_response
        # final_ans = extract_final_answer(ans)
        return final_ans, all_messages
