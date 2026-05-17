# -*- coding: utf-8 -*-

from input_instance import InputInstance
import csv
import os
import json


class CSVDataReader:
    """
    Reads in the CSV-format dataset.
     Each line contains several sentences
     (utterance_1, utterance_2) and 2 labels (label_1, label_2)
    """

    def __init__(self, dataset_folder, delimiter="\t",
                 quoting=csv.QUOTE_NONE):
        '''
        the parameters denote the index number of different properties
        '''
        self.dataset_folder = dataset_folder
        self.delimiter = delimiter
        self.quoting = quoting

        self.label2idx_test = self.convert_emotion_json(os.path.join(self.dataset_folder, 'test_label.json'))
        self.label2idx_valid = self.convert_emotion_json(os.path.join(self.dataset_folder, 'valid_label.json'))
        self.label2idx_train = self.convert_emotion_json(os.path.join(self.dataset_folder, 'see_relation.json'))

    def convert_emotion_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        result = {}
        for k, (key, value) in enumerate(data.items()):
            emotion = value.split(':')[0].strip().lower()
            result[emotion] = int(k)
        return result

    def get_instances(self, filename, max_instances=0):
        """
        filename specified which data
         split to use (train.csv, dev.csv, test.csv).
        """
        data = csv.reader(
            open(os.path.join(self.dataset_folder, filename),
                 encoding="utf-8"),
            delimiter=self.delimiter, quoting=self.quoting)
        instances = []
        next(data)
        for id, row in enumerate(data):
            uttrances = row[0]
            labels = row[1]
            if filename == 'train_see.csv':
                labels = self.label2idx_train[labels.lower()]
            elif filename == 'test.csv':
                labels = self.label2idx_test[labels.lower()]
            elif filename == 'valid.csv':
                labels = self.label2idx_valid[labels.lower()]
            else:
                print('Error')
            instances.append(InputInstance(
                guid=str(id),
                texts=uttrances,
                labels=labels))
            if max_instances > 0 and len(instances) >= max_instances:
                break
        return instances