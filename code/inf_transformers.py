import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
import argparse
import warnings
from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import EarlyStoppingCallback
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from safetensors.torch import load_file

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--output_path", type=str, default="vllm_results_14b")
parser.add_argument("--model_path", type=str, default="../DeepSeek-R1-Distill-Llama-8B")
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--save_steps", type=int, default=200)
parser.add_argument("--lr_scheduler_type", type=str, default="linear")
parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
parser.add_argument("--total_batch_size", type=int, default=256)
parser.add_argument("--train_size", type=int, default=256)
parser.add_argument("--val_size", type=int, default=1000)
parser.add_argument("--resume_from_checkpoint", type=str, default="lora_llama/Llama3.1_ml-1m_mixed_text_new")
parser.add_argument("--lora_remote_checkpoint", type=str, default=None)
parser.add_argument("--ignore_data_skip", type=str, default="False")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--only_test", action='store_true')
parser.add_argument("--use_lora", type=int, default=0)
parser.add_argument("--dataset", type=str, default='BookCrossing')
parser.add_argument("--K", type=int, default=30)
parser.add_argument("--temp_type", type=str, default="rerank")
parser.add_argument("--emb_type", type=str, default="mix3")
parser.add_argument("--sim_user", type=bool, default=False)
parser.add_argument("--train_type", type=str, default="mixed")
parser.add_argument("--postfix", type=str, default="2")

args = parser.parse_args()
print("\n")
print('*'*50)
print(args)
print('*'*50)
print("\n")

assert args.dataset in ['ml-1m', 'BookCrossing', 'GoodReads', 'amazon-movies']
data_path = f"data/{args.dataset}/proc_data/data"

transformers.set_seed(args.seed)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
if not args.wandb:
    os.environ["WANDB_MODE"] = "disable"

MICRO_BATCH_SIZE = 4
BATCH_SIZE = min(args.total_batch_size, args.train_size)
MAX_STEPS = None
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 5
LEARNING_RATE = args.lr
CUTOFF_LEN = 2048
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = args.val_size
USE_8bit = False

if USE_8bit is True:
    warnings.warn("If your version of bitsandbytes>0.37.2, Please downgrade bitsandbytes's version, for example: pip install bitsandbytes==0.37.2")
        
TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

if args.sim_user:
    fp = '/'.join([data_path, f"test/test_{args.K}_{args.temp_type}_{args.emb_type}_sim.json"])
else:
    fp = '/'.join([data_path, f"test/test_{args.K}_{args.temp_type}_{args.emb_type}.json"])

DATA_PATH = {
    "test": fp
}

OUTPUT_DIR = args.output_path

device_map = "auto"

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    load_in_8bit=USE_8bit,
    device_map=device_map,
)
tokenizer = AutoTokenizer.from_pretrained(
    args.model_path, add_eos_token=True, padding_side="left",
)
tokenizer.pad_token_id = 0
print("Model loaded.")

if args.use_lora:
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    print("Load lora weights...")
    adapters_weights = load_file(os.path.join(args.resume_from_checkpoint, "adapter_model.safetensors"))
    set_peft_model_state_dict(model, adapters_weights)
    print("lora load results.")

data = load_dataset("json", data_files=DATA_PATH)
print("Data loaded.")

test_data = data['test']

model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

def compute_metrics(preds, golds):
    auc = roc_auc_score(golds, preds)
    logloss = log_loss(golds, preds)
    acc = accuracy_score(golds, [pred > 0.5 for pred in preds])
    return {
        'auc': auc, 
        'logloss': logloss, 
        'acc': acc, 
    }

print("Evaluate on the test set...")
if args.dataset=='BookCrossing':
    eval_size = len(test_data)
else:
    eval_size = 10000

results = []
preds = []
golds = [int(data_point['output'] == 'Yes.') for data_point in test_data][:eval_size]
cnt = 0

with open(f"{args.output_path}/{args.dataset}/vllm_{args.K}_{args.temp_type}_{args.emb_type}_{args.postfix}.json", "r") as f:
    reasong_data = json.load(f)["samples"]

for data_point in tqdm(test_data):
    if cnt >= eval_size:
        break
    
    if "</think>" in reasong_data[cnt]["pred"]:
        reasong_data_split = reasong_data[cnt]["pred"].split("</think>")
        longest_path = max(reasong_data_split, key=len)
        local_reasoning_data = "\n<think>" + longest_path + "\n</think>\n\n"
    else:
        local_reasoning_data = "\n<think>" + reasong_data[cnt]["pred"] + "\n</think>\n\n"

    if args.model_path == '../llama/Llama-3.1-8B-Instruct':
        local_reasoning_data=''

    messages = [
    {
        "role": "system",
        "content": "",
    },
    {
        "role": "user", 
        "content": data_point['input']+local_reasoning_data
    }
]   
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs, return_tensors="pt", add_special_tokens=False)
    inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
    outputs = model.generate(
        **inputs, 
        temperature=0.6,
        do_sample=True,
        max_new_tokens=2,
        no_repeat_ngram_size=2,
        return_dict_in_generate=True,
        output_logits=True,
        pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs.sequences

    if not ('Deep' in args.model_path and 'Lla' in args.model_path):
        logits = outputs.logits[0].softmax(dim=-1) 
    else:
        logits = outputs.logits[1].softmax(dim=-1) 

    if not 'Qwen' in args.model_path:
        binary_logits = logits[:,[9642, 2822]].clone().detach().softmax(dim=-1)
    else:
        binary_logits = logits[:,[9454, 2753]].clone().detach().softmax(dim=-1)

    print(binary_logits)

    prediction = tokenizer.batch_decode(generated_ids[:, inputs['input_ids'].size(1):], skip_special_tokens=True)[0]

    preds.extend(binary_logits[:, 0].tolist())

    results.append({'input': data_point['input'], 'output': data_point['output'], 'pred': prediction})

    cnt += 1

test_metrics = compute_metrics(preds, golds)
print(test_metrics)

if args.model_path == '../DeepSeek-R1-Distill-Llama-8B':
    with open(f'{args.output_path}/{args.dataset}/vllm_{args.K}_{args.temp_type}_{args.emb_type}_{args.postfix}_dsllama8b.txt', 'w') as file:
        for item in preds:
            file.write(f"{item}\n")
elif args.model_path == '../llama/Llama-3.1-8B-Instruct':
    with open(f'{args.output_path}/{args.dataset}/vllm_{args.K}_{args.temp_type}_{args.emb_type}_{args.postfix}_{args.resume_from_checkpoint[19:].replace("/", "")}.txt', 'w') as file:
        print(f'{args.output_path}/{args.dataset}/vllm_{args.K}_{args.temp_type}_{args.emb_type}_{args.postfix}_{args.resume_from_checkpoint[19:].replace("/", "")}.txt')
        for item in preds:
            file.write(f"{item}\n")
else:
    with open(f'{args.output_path}/{args.dataset}/vllm_{args.K}_{args.temp_type}_{args.emb_type}_{args.postfix}_other.txt', 'w') as file:
        for item in preds:
            file.write(f"{item}\n")

torch.cuda.empty_cache()