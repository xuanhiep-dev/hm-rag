from collections import Counter
from langchain_community.llms.ollama import Ollama
import re
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import random
import os

class SummaryAgent:
    def __init__(self, config):
        self.config = config
        self.text_llm = Ollama(base_url="http://localhost:11434", model="qwen2.5:14b")
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto")

    def summarize(self, problems, shot_qids, qid, cur_ans) -> str:
        problem = problems[qid]
        question = problem['question']
        choices = problem["choices"]
        answer = problem['answer']
        image = problem['image']
        caption = problem['caption']
        split = problem["split"]
        
        most_ans = self.get_most_common_answer(cur_ans)   
        # Placeholder for summarization logic
        if len(most_ans) == 1:
            prediction = self.get_result(most_ans[0])  # 'A', ..., 'E'
            pred_idx = self.get_pred_idx(prediction, choices, self.config.options)
        else:
            if image == "image.png":
                image_path = os.path.join(self.config.image_root, split, qid, image)
            else:
                image_path = ""
            output_text = cur_ans[0]
            output_graph = cur_ans[1]
            output_web = cur_ans[2]
            output = self.refine(output_text, output_graph, output_web, 
                                 problems, shot_qids, qid, self.config, image_path)
            if output == None:
                output  = "FAILD"
            print(f"output: {output}")

            ans_fusion = self.get_result(output)
            pred_idx = self.get_pred_idx(ans_fusion, choices, self.config.options)
        return pred_idx, cur_ans
    
    def get_most_common_answer(slef, res):
        """
        Get the most common answer from the list of answers
        """
        counter = Counter(res)

        # 获取最高频率
        max_count = max(counter.values())

        # 收集所有频率等于 max_count 的值
        most_common_values = [item for item, count in counter.items() if count == max_count]
        return most_common_values
    
    def refine(self, output_text, output_graph, output_web, problems, shot_qids, qid, args, image_path):
    
        prompt = build_prompt(problems, shot_qids, qid, args)
        # prompt = f"{prompt} \n Output1: {output_text}. \n Output2: {output_graph} \n Output3: {output_web}. Summary the outputs with  chain-of-thought with format 'Answer: The answer is A, B, C, D, E or FAILED. \n BECAUSE: '"
        prompt = f"{prompt} The answer is A, B, C, D, E or FAILED. \n BECAUSE: "
        if not image_path:
            output = self.text_llm.invoke(prompt)
    
        else:
            output = self.qwen_reasoning(prompt, image_path)
            print(f"**** output: {output}")
            output = self.text_llm.invoke(f"{output[0]} Summary the above information with format 'Answer: The answer is A, B, C, D, E or FAILED.    \n BECAUSE: '")

        return output
    
    def get_result(self, output):
        # extract the answer
        # pattern = re.compile(r'The answer is ([A-Z]).')
        pattern = re.compile(r'The answer is ([A-E])')
        res = pattern.findall(output)
        if len(res) == 1:
            answer = res[0]  # 'A', 'B', ...
        else:
            answer = "FAILED"

        return answer

    def get_pred_idx(self, prediction, choices, options):
        """
        Get the index (e.g. 2) from the prediction (e.g. 'C')
        """
        if prediction in options[:len(choices)]:
            return options.index(prediction)
        else:
            return random.choice(range(len(choices)))


    def qwen_reasoning(self, prompt, image_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
