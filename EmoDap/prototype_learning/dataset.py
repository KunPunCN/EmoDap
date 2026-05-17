import re
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModel
import os
from input_instance import InputInstance
from csv_reader import CSVDataReader


def inverse_tokenize(tokens):
    r"""
    Convert tokens to sentence.
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    Watch out!
    Default punctuation add to the word before its index,
    it may raise inconsistency bug.
    :param list[str]r tokens: target token list
    :return: str
    """
    assert isinstance(tokens, list)
    text = ' '.join(tokens)
    step1 = text.replace("`` ", '"') \
        .replace(" ''", '"') \
        .replace('. . .', '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
        "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    step7 = step6.replace('do nt', 'dont').replace('Do nt', 'Dont')
    step8 = step7.replace(' - ', '-')
    return step8.strip()

def mark_fewrel_entity(new_pos, new_entity_h, new_entity_t, sent_len):
    mark_head = np.array([0] * sent_len) 
    mark_tail = np.array([0] * sent_len)
    mark_head[new_entity_h[0]:new_entity_h[1]] = 1 # mark head entity, which is between [E1] and [E1/]
    mark_tail[new_entity_t[0]:new_entity_t[1]] = 1 # mark head entity, which is between [E2] and [E2/]
    marked_e1 = np.array([0] * sent_len)
    marked_e2 = np.array([0] * sent_len)
    marked_e1[new_pos[0]] = 1 # mark [E1]
    marked_e2[new_pos[1]] = 1 # mark [E2]
    return torch.tensor(marked_e1), torch.tensor(marked_e2), \
             torch.tensor(mark_head), torch.tensor(mark_tail)

def pad_or_truncate(tensor, target_width):
    current_width = tensor.size(0)
    if current_width < target_width:
        pad_size = target_width - current_width
        padding = torch.zeros(pad_size, dtype=tensor.dtype, device=tensor.device)
        padded_tensor = torch.cat((tensor, padding), dim=0)
        return padded_tensor
    elif current_width > target_width:
        truncated_tensor = tensor[:target_width]
        return truncated_tensor
    else:
        return tensor

class Dataset(Dataset):

    def __init__(self, mode, dataset_path, m, pretrained_model_name_or_path, max_len, model, args, use_mlm = False, expand_or_not = True):
        '''
        data_file: dataset path
        description_file: relation description file path
        description_file_processed: RE-matching description file path
        m: the number of unseen relations
        '''
        super(Dataset, self).__init__()
        self.data_types = ["train", "dev", "test"]
        assert mode in self.data_types
        self.mode = mode # train, dev, test

        csvDataReader = CSVDataReader(dataset_path)
        self.data = {}  # {"train": [sample],...}
        self.data['train'] = csvDataReader.get_instances('train_see.csv')  # guid, texts, labels
        self.data['dev'] = csvDataReader.get_instances('valid.csv')
        self.data['test'] = csvDataReader.get_instances('test.csv')

        self.description_file = os.path.join(dataset_path, 'see_relation.json')
        self.description_file2 = os.path.join(dataset_path, 'valid_label.json')
        self.description_file3 = os.path.join(dataset_path, 'test_label.json')

        self.m = m
        self.pretrained_model_name_or_path = pretrained_model_name_or_path # bert-base-uncased or others
        self.max_len = max_len # seq max length
        self.use_mlm = use_mlm # use mlm expend entitys or not
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path) # "../bert-base-uncased"
        # self.tokenizer = AutoTokenizer.from_pretrained('deberta-v3-large')
        self.label_ids = {} # {"train":[str,...],"dev":[str,...],"test":[str,...],}
        self.descriptions = {} # {"train": [sample],...}


        self.data_features = {}
        self.des_features = {}

        self.head_mark_ids = 1001
        self.tail_mark_ids = 1030
        self.model = model
        self.args = args

        self.read_data()
        if(expand_or_not):
            self.expand_data()
        # self.convert_data_and_des_to_features()
        self.convert_data_and_des_to_features_DSSM()

    
    def read_data(self):
        # load relation_description
        for (file, t) in zip([self.description_file, self.description_file2, self.description_file3], ['train', 'dev', 'test']):
            with open(file, 'r', encoding='utf-8') as rd:
                relation_desc = json.load(rd)
                relation = {}
                description, input_ids, attention_mask = [],[],[]
                for i in relation_desc.values():
                    tokens_info = self.tokenizer(i)
                    des_input_ids = torch.tensor(tokens_info['input_ids'])
                    des_attention_mask = torch.tensor(tokens_info['attention_mask'])
                    description.append(i)
                    input_ids.append(pad_or_truncate(des_input_ids, self.max_len))
                    attention_mask.append(pad_or_truncate(des_attention_mask, self.max_len))

                relation['description'] = description
                relation['des_input_ids'] = input_ids
                relation['des_attention_mask'] = attention_mask
                self.descriptions[t] = relation

        print(f'train data numbers: {len(self.data["train"])}')
        print(f'dev data numbers: {len(self.data["dev"])}')
        print(f'test data numbers: {len(self.data["test"])}')

    def convert_data_and_des_to_features_DSSM(self):
        # convert data to features
        for t in self.data_types:
            data = self.data[t]#guid, texts, labels
            self.data_features[t] = []
            for sample in tqdm(data, "convert data to features: "):
                # guid = sample.guid
                sentence = sample.texts
                label = sample.labels
                input_idss, attention_masks, rid_tensor  = [],[],[]

                tokens_info = self.tokenizer(sentence)
                input_ids = torch.tensor(tokens_info['input_ids'])
                attention_mask = torch.tensor(tokens_info['attention_mask'])
                input_idss.append(pad_or_truncate(input_ids, self.max_len))
                attention_masks.append(pad_or_truncate(attention_mask, self.max_len))
                rid_tensor.append(int(label))

                input_idss = torch.stack(input_idss)#.view(-1)#[b,128] ->[b*128]
                attention_masks = torch.stack(attention_masks)#.view(-1)
                rid_tensor = torch.tensor(rid_tensor)
                sample_features = {
                    "input_ids": input_idss,
                    "attention_mask": attention_masks,
                    "rid": rid_tensor,
                }
                self.data_features[t].append(sample_features)
                
        # convert descriptions to features
        for t in self.data_types:
            self.des_features[t] = []
            des = self.descriptions[t]
            for rid, (input_ids, attention_mask) in enumerate(zip(des["des_input_ids"],des["des_attention_mask"])):
                self.des_features[t].append(torch.cat((input_ids, attention_mask), dim=0))


    def __getitem__(self, index):
            return self.data_features[self.mode][index]

    
    def get_evaluate_des_features(self):
        return torch.stack(self.des_features[self.mode])

    def get_source_des_features(self):
        return torch.stack(self.des_features['train'])
    
    def convert_rid_to_label(self, rid):
        return self.label_ids[self.mode].index(rid)
    
    def __len__(self):
        return len(self.data[self.mode])