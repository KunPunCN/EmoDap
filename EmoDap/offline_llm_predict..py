from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import torch
import transformers
import re
from sklearn.metrics import f1_score, accuracy_score

import os
import json

from collections import defaultdict
import random
import time
from utils import convert_emotion_json

zero_shot_setting = False
semantic_top = True
##########settings
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model_dir = '/your_path/Qwen2.5-7B-Instruct'
dataset_name = 'EDOS'
run_seed = 1
base_dir = os.path.join('data', dataset_name, 'random_splits', 'fold_{}'.format(run_seed))
data_dir = os.path.join(base_dir, 'test.csv')
label_dir = os.path.join(base_dir, 'test_label.json')

topk=6
if dataset_name == 'EDOS':
    topk = 8
if dataset_name == 'GE':
    topk = 5
candi_dir = os.path.join(base_dir, f'{dataset_name.lower()}_semantic_top{topk}.txt')


if zero_shot_setting == True:
    save_file_name = 'llamb_zero_shot.txt'
else:
    save_file_name = 'llamb_four_shot.txt'


if 'Qwen' in model_dir:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_dir,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=device,
    )  # "auto"
    terminators = []
    if tokenizer.eos_token_id is not None:
        terminators.append(tokenizer.eos_token_id)

    if hasattr(tokenizer, "eot_id") and tokenizer.eot_id is not None:
        terminators.append(tokenizer.eot_id)
    elif tokenizer.convert_tokens_to_ids("<|eot_id|>") != tokenizer.unk_token_id:
        term_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if term_id is not None:
            terminators.append(term_id)

    if not terminators:
        terminators = [tokenizer.eos_token_id] if tokenizer.eos_token_id else [2]
else:
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_dir,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=device,
    )
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    if pipeline.tokenizer.pad_token_id is None:
        pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id


data = []
label = []
label_texts = defaultdict(list)
with open(data_dir, 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file, delimiter='\t')
    next(csv_reader)
    for row in csv_reader:
        data.append(row[0])
        label.append(row[1])
        label_texts[row[1]].append(row[0])


with open(candi_dir, 'r', encoding='utf-8') as file:
    candites = file.readlines()
candites = [i.strip().split(', ') for i in candites]

assert len(label) == len(candites)


label2idx = convert_emotion_json(label_dir)
label_list = list(label2idx.keys())

with open(os.path.join('data', dataset_name, 'uns_candis.json'), 'r', encoding='utf-8') as f:
    sentence_dict = json.load(f)

x = 0
y = []
pred = []
for a,b in zip(label, candites):
    candi_id = [int(k) for k in b]
    candi = [label_list[k] for k in candi_id]



start_time = time.time()
with open(os.path.join(base_dir, save_file_name), 'w', encoding='utf-8') as new_file_obj:
    for i in range(len(data)):
        sentence = data[i]
        candi_id = [int(k) for k in candites[i]]
        candi = [label_list[k] for k in candi_id]

        if zero_shot_setting == True:
            if semantic_top == False:
                promtp_t = str(label_list)
            else:
                promtp_t = str(candi)
            s1 = 'Given a sentence, please determine the emotion it conveys.\n' \
                 'Sentence:' + sentence + '\nChoose your answer from ' + promtp_t + '. Don’t explain yourself.'
        else:
            if semantic_top == False:
                candi_list = label_list
                promtp_t = str(label_list)
            else:
                candi_list = candi
                promtp_t = str(candi)
            d = ''
            for ids, j in enumerate(candi_list):
                d = d + '\nEmotion ' + str(ids + 1) + ': ' + j + '\n Demonstrations: '
                x_dict = sentence_dict[str(label2idx[j])]
                for kk in range(len(x_dict)):
                    d = d + '\n' + x_dict[kk]
                    if kk==0:
                        break
            s1 = 'Given a sentence, please determine the emotion it conveys.\nHere are demonstrations of these emotions:' + d + \
                 '\nPlease understand the meaning of each emotion through these demonstrations.' \
                 'Sentence:' + sentence + '\nChoose your answer from ' + promtp_t + '. Don’t explain yourself, just give me one word.'


        messages = [
            {"role": "system", "content": "You are a psychology expert."},
            {"role": "user", "content": s1},
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=32,
            eos_token_id=terminators,
            pad_token_id=128009,
            # pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            # temperature=0.6,
            # top_p=0.9,
        )
        responsex = outputs[0]["generated_text"][-1]['content'].lower()
        response = responsex.strip('\'').strip('\'')
        if ': ' in response:
            response = response.split(": ")[1]
        if response in candi:
            new_file_obj.write(response)
            new_file_obj.write('\n')
        else:
            new_file_obj.write('caring')
            new_file_obj.write('\n')

        if response in candi:
            pred.append(response)
        else:
            pred.append('None')
            y.append(i)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time: {elapsed_time:.4f} sec")

weighted_f1 = f1_score(label, pred, average='weighted')
macro_f1 = f1_score(label, pred, average='macro')
accuracy = accuracy_score(label, pred)
print(f"Weighted F1 Score: {weighted_f1}, Macro F1: {macro_f1}, Acc: {accuracy}")




