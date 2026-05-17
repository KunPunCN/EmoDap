import os
import json
from sklearn.metrics import f1_score, accuracy_score
from openai import OpenAI
import csv
import json
import time
import ast
from anthropic import Anthropic
from utils import convert_emotion_json

api_base = "your_api_base"
api_key = "your_api_key"
dataset_name = 'EDOS'
run_seed = 1
base_dir = os.path.join('data', dataset_name, 'random_splits', 'fold_{}'.format(run_seed))
data_dir = os.path.join(base_dir, 'test.csv')
label_dir = os.path.join(base_dir, 'test_label.json')

client = OpenAI(
    api_key=api_key,
    base_url=api_base
)
# client = Anthropic(
#     base_url=api_base,
#     api_key=api_key,
# )


def chat_with_gpt(client, messages):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        max_tokens=3000,
        temperature=0.4,
        top_p=0.9
    )
    return response.choices[0].message.content

def chat_with_claude(client, messages):
    response = client.messages.create(
        max_tokens=1024,
        messages=messages,
        model="claude-3-7-sonnet-20250219",
    )
    return response.content[0].text




data = []
label = []
with open(data_dir, 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file, delimiter='\t')
    next(csv_reader)
    for row in csv_reader:
        data.append(row[0])
        label.append(row[1].lower())



label2idx = convert_emotion_json(label_dir)
print(label2idx)
label_list = list(label2idx.keys())


x = 0
y = []
pred = []



def process_string(s):
    count = s.count("'")
    if count >= 2:
        first = s.find("'")
        second = s.find("'", first + 1)
        return s[first+1:second]
    return s

with open(os.path.join(base_dir, 'deepseek_zero_shot.txt'), 'w', encoding='utf-8') as new_file_obj:
    for i in range(len(data)):
        sentence = data[i]

        promtp_t = str(label_list)
        s1 = 'Given a sentence, please determine the emotion it conveys.\n' \
             'Sentence:' + sentence + '\nChoose your answer from ' + promtp_t + '. Don’t explain yourself.'

        messages = [
            {"role": "system", "content": "You are a psychology expert."},
            {"role": "user", "content": s1},
        ]

        response = chat_with_gpt(client, messages)
        response = process_string(response)

        if response in label_list:
            pred.append(response)
        else:
            print(response)
            pred.append('xxx')
            y.append(i)


weighted_f1 = f1_score(label, pred, average='weighted')
macro_f1 = f1_score(label, pred, average='macro')
accuracy = accuracy_score(label, pred)
print(f"Weighted F1 Score: {weighted_f1}, Macro F1: {macro_f1}, Acc: {accuracy}")
