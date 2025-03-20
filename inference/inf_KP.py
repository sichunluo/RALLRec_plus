import os
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

parser = argparse.ArgumentParser()

parser.add_argument("--output_path", type=str, default="vllm_results_hint")
parser.add_argument("--model_path", type=str, default="../DeepSeek-R1-Distill-Llama-8B")
parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
parser.add_argument("--total_batch_size", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default='BookCrossing')

parser.add_argument("--K", type=int, default=30)
parser.add_argument("--temp_type", type=str, default="rerank")
parser.add_argument("--emb_type", type=str, default="mix3")
parser.add_argument("--sim_user", type=bool, default=False)

parser.add_argument("--num_anmwer", type=int, default=6)

args = parser.parse_args()

num_anmwer = args.num_anmwer

print("Initializing vLLM engine...")
llm = LLM(
    model=args.model_path,
    tokenizer=args.model_path,
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.85,
    trust_remote_code=True
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
yes_ids = tokenizer("Yes", add_special_tokens=False)["input_ids"]
no_ids = tokenizer("No", add_special_tokens=False)["input_ids"]

sampling_params = SamplingParams(
    temperature=0.6,
    max_tokens=4800,
    n=num_anmwer,
    stop_token_ids=[tokenizer.eos_token_id],
    skip_special_tokens=True,
    seed=args.seed
)

data_path = f"data/{args.dataset}/proc_data/data"
fp = f"{data_path}/test/test_{args.K}_{args.temp_type}_{args.emb_type}_sim.json" if args.sim_user \
    else f"{data_path}/test/test_{args.K}_{args.temp_type}_{args.emb_type}.json"

test_data = load_dataset("json", data_files={"test": fp})["test"]

with open(f'vllm_results/{args.dataset}/vllm_30_rerank_mix3_1__{args.dataset}_mixed_text_new.txt', 'r') as f:
    llama_pred = [float(line.strip()) for line in f]

def format_prompt(data_point, llama_pred):
    if llama_pred>0.5:
        label = 'Yes'
    else:
        label = 'No'
    
    if abs(llama_pred-0.5)>0.2:
        ex= 'is'
    elif abs(llama_pred-0.5)>0.1:
        ex= 'may be'
    else:
        ex= 'might be'
    
    hint=f"\nAnother one think the answer "+ex+" "+label+f".\n"

    return tokenizer.apply_chat_template([
        {"role": "system", "content": ""},
        {"role": "user", "content": data_point["input"] +hint+ "\n<think>"}
    ], tokenize=False)

if args.dataset == 'BookCrossing':
    prompts = [format_prompt(test_data[d], llama_pred[d]) for d in range(len(test_data))]
else:
    prompts = [format_prompt(test_data[d], llama_pred[d]) for d in range(10000)]

def extract_probability(output):
    text = output.outputs[0].text.strip()
    if "</think>" in text:
        after_think = text.split("</think>")[-1].strip().lower()
        if "yes" in after_think:
            return 1.0
        elif "no" in after_think:
            return 0.0
    else:
        last_chars = text[-20:].strip().lower()
        if "yes" in last_chars:
            return 1.0
        elif "no" in last_chars:
            return 0.0
    return 0.5

results = []
raw_results = []
for i in range(16):
    raw_results.append([])

batch_size = 100
total_num = 10000

print("Starting inference...")

import time 
start_time = time.time()
for i in tqdm(range(0, total_num, batch_size), desc="Processing"):
    if i+batch_size>len(test_data):
        batch_prompts = prompts[i:len(test_data)]
    else:
        batch_prompts = prompts[i:i+batch_size]

    outputs = llm.generate(batch_prompts, sampling_params)

    for output in outputs:
        results.append(extract_probability(output))
        for i in range(num_anmwer):
            raw_results[i].append(output.outputs[i].text.strip())

total_time = time.time() - start_time

golds = [int(d["output"] == "Yes.") for d in test_data][:(total_num)]
metrics = {
    "auc": roc_auc_score(golds, results),
    "logloss": log_loss(golds, results),
    "accuracy": accuracy_score(golds, [p > 0.5 for p in results])
}

print("\nEvaluation Results:")
print(json.dumps(metrics, indent=2))

os.makedirs(f"{args.output_path}/{args.dataset}", exist_ok=True)

def save_file_with_incremented_name(raw_results):
    counter = 1
    file_name = f"{args.output_path}/{args.dataset}/vllm_{args.K}_{args.temp_type}_{args.emb_type}_{counter}.json"
    
    while os.path.exists(file_name):
        counter += 1
        file_name = f"{args.output_path}/{args.dataset}/vllm_{args.K}_{args.temp_type}_{args.emb_type}_{counter}.json"
        
    with open(f"{args.output_path}/{args.dataset}/vllm_{args.K}_{args.temp_type}_{args.emb_type}_{counter}.json", "w") as f:
        json.dump({
            "args": vars(args),
            "metrics": metrics,
            'total_time':total_time,
            "samples": [{
                "input": test_data[i]["input"],
                "output": test_data[i]["output"],
                "pred": raw_results[i]
            } for i in range(len(results))]
        }, f, indent=2)
    
    return file_name

for i in range(num_anmwer):
    save_file_with_incremented_name(raw_results[i])

print(f"File Saved")