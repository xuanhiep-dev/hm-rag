from collections import Counter
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import re
import random
import os


class SummaryAgent:
    def __init__(self, config):
        self.config = config

        # Dùng Qwen-VL làm cả mô hình tóm tắt văn bản + xử lý hình ảnh
        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    # ===============================
    # Summarization pipeline
    # ===============================
    def summarize(self, problems, shot_qids, qid, cur_ans) -> str:
        problem = problems[qid]
        question, choices = problem["question"], problem["choices"]
        answer, image, caption, split = (
            problem["answer"],
            problem["image"],
            problem["caption"],
            problem["split"],
        )

        most_ans = self.get_most_common_answer(cur_ans)

        if len(most_ans) == 1:
            prediction = self.get_result(most_ans[0])
            pred_idx = self.get_pred_idx(
                prediction, choices, self.config.options)
        else:
            image_path = (
                os.path.join(self.config.image_root, split, qid, image)
                if image and image != "image.png"
                else ""
            )

            output_text = cur_ans[0] if len(
                cur_ans) > 0 else "Không có kết quả Vector."
            output_graph = cur_ans[1] if len(
                cur_ans) > 1 else "Không có kết quả Graph."
            output_web = cur_ans[2] if len(
                cur_ans) > 2 else "Không có kết quả Web."

            output = self.refine(
                output_text, output_graph, output_web, problems, shot_qids, qid, self.config, image_path
            )
            output = output or "FAILED"
            print(f"[SummaryAgent] output: {output}")

            ans_fusion = self.get_result(output)
            pred_idx = self.get_pred_idx(
                ans_fusion, choices, self.config.options)

        return pred_idx, cur_ans

    # ===============================
    # Helper functions
    # ===============================
    def get_most_common_answer(self, res):
        counter = Counter(res)
        max_count = max(counter.values())
        return [k for k, v in counter.items() if v == max_count]

    def refine(self, output_text, output_graph, output_web, problems, shot_qids, qid, args, image_path):
        # ---- Build prompt ----
        prompt = f"""
Câu hỏi: {problems[qid]['question']}
Các câu trả lời có thể: {problems[qid]['choices']}
Dưới đây là ba câu trả lời từ các tác nhân khác nhau:
1. Vector Retrieval: {output_text}
2. Graph Retrieval: {output_graph}
3. Web Retrieval: {output_web}
Hãy hợp nhất thông tin này và chọn đáp án đúng nhất (A,B,C,D,E hoặc FAILED).
Giải thích ngắn gọn: 
"""
        # ---- Text-only mode ----
        if not image_path:
            return self.qwen_generate(prompt)

        # ---- Vision-Language reasoning ----
        output = self.qwen_reasoning(prompt, image_path)
        return output[0] if isinstance(output, list) else output

    def qwen_generate(self, text_prompt: str):
        # Dùng Qwen-VL nhưng chỉ xử lý text
        messages = [{"role": "user", "content": [
            {"type": "text", "text": text_prompt}]}]
        txt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[txt], padding=True, return_tensors="pt").to(self.model.device)
        out_ids = self.model.generate(**inputs, max_new_tokens=1024)
        trim = [out[len(inp):] for inp, out in zip(inputs.input_ids, out_ids)]
        decoded = self.processor.batch_decode(trim, skip_special_tokens=True)
        return decoded[0]

    def get_result(self, output):
        pattern = re.compile(r"[Aa]nswer\s*is\s*([A-E])")
        res = pattern.findall(output)
        return res[0] if len(res) == 1 else "FAILED"

    def get_pred_idx(self, prediction, choices, options):
        return options.index(prediction) if prediction in options[: len(choices)] else random.randrange(len(choices))

    def qwen_reasoning(self, prompt, image_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)
        out_ids = self.model.generate(**inputs, max_new_tokens=2048)
        trim = [out[len(inp):] for inp, out in zip(inputs.input_ids, out_ids)]
        decoded = self.processor.batch_decode(trim, skip_special_tokens=True)
        return decoded
