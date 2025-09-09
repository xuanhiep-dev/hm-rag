import os
import re
import json
import argparse
import random
from tqdm import tqdm
import sys
from agents.multi_retrieval_agents import MRetrievalAgent

import openai
from openai import OpenAI
from langchain_community.llms.ollama import Ollama

import os
os.environ["http_proxy"] = "http://127.0.0.1:11434"
os.environ["https_proxy"] = "http://127.0.0.1:11434"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--image_root', type=str)
    parser.add_argument('--output_root', type=str)
    parser.add_argument('--caption_file', type=str)
    parser.add_argument('--model', type=str, default='gpt3')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    # user options
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'val', 'minival'])
    parser.add_argument('--prompt_format',
                        type=str,
                        default='CQM-A',
                        choices=[
                            'CQM-A', 'CQM-LA', 'CQM-EA', 'CQM-LEA', 'CQM-ELA', 'CQM-AL', 'CQM-AE', 'CQM-ALE', 'QCM-A',
                            'QCM-LA', 'QCM-EA', 'QCM-LEA', 'QCM-ELA', 'QCM-AL', 'QCM-AE', 'QCM-ALE', 'QCML-A', 'QCME-A',
                            'QCMLE-A', 'QCLM-A', 'QCEM-A', 'QCLEM-A', 'QCML-AE'
                        ],
                        help='prompt format template')
    # VectorRetrieval settings
    parser.add_argument('--working_dir', type=str)
    parser.add_argument('--llm_model_name', type=str, default='qwen2.5:7b')
    parser.add_argument('--mode', type=str, default='hybrid')
    parser.add_argument('--serper_api_key', type=str)
    parser.add_argument('--top_k', type=int, default=4)
    # GPT settings
    parser.add_argument('--openai_key', type=str)
    parser.add_argument('--engine', type=str, default='gpt-4o')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')

    args = parser.parse_args()
    return args

def load_data(args):
    problems = json.load(open(os.path.join(args.data_root, 'problems.json')))
    pid_splits = json.load(open(os.path.join(args.data_root, 'pid_splits.json')))
    captions = json.load(open(args.caption_file))["captions"]

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    qids = pid_splits['%s' % (args.test_split)]
    qids = qids[:args.test_number] if args.test_number > 0 else qids
    print(f"number of test problems: {len(qids)}\n")

    # pick up shot examples from the training set
    shot_qids = args.shot_qids
    train_qids = pid_splits['train']
    if shot_qids == None:
        assert args.shot_number >= 0 and args.shot_number <= 32
        shot_qids = random.sample(train_qids, args.shot_number)  # random sample
    else:
        shot_qids = [str(qid) for qid in shot_qids]
        for qid in shot_qids:
            assert qid in train_qids  # check shot_qids
    print("training question ids for prompting: ", shot_qids, "\n")

    return problems, qids, shot_qids

def main():
    args = parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)

    problems, qids, shot_qids = load_data(args)  # probelms, test question ids, shot example ids

    result_file = args.output_root + '/' + args.label + '_' + args.test_split + '.json'
    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    sum_agent = MRetrievalAgent(args)
    correct = 0
    results = {}
    outputs = {}

    failed = []
    # for qid in tqdm(qids):
    for i, qid in enumerate(qids):
        if args.debug and i > 10:
            break
        if args.test_number > 0 and i >= args.test_number:
            break

        problem = problems[qid]
 
        answer = problem['answer']
        final_ans, all_messages = sum_agent.predict(problems, shot_qids, qid)
        outputs[qid] = all_messages
        results[qid] = final_ans
        if final_ans == answer:
            correct += 1
        else:
            failed.append(qid)
        if (i + 1) % args.save_every == 0:
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {result_file} after {i + 1} examples.")

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {result_file} after {len(qids)} examples.")
    print(f"Number of correct answers: {correct}/{len(qids)}")
    print(f"Accuracy: {correct / len(qids):.4f}")
    print(f"Failed question ids: {failed}")
    print(f"Number of failed questions: {len(failed)}")

if __name__ == "__main__":
    main()


