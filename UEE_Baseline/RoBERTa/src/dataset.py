import torch
from torch.utils.data import Dataset
import json

class UEEDataset(Dataset):
    def __init__(self, json_list, tokenizer, max_length):
        self.json_list = json_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def getJsonList(self):
        return self.json_list

    def __getitem__(self, idx):
        dic = self.json_list[idx]
        text1 = dic['paraTxt']

        text2_li = list()
        for cp in dic['claims']:
            li = cp['claimTxt']
            text2_li.extend(li)
        text2 = ''.join(text2_li)

        is_label_present = 'label' in dic
        if is_label_present:
            label = 1 if dic['label'] == 'positive' else 0

        encoding = self.tokenizer(
                text1,
                text2,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                )

        encoding['position_ids'] = list(range(self.max_length))

        encoding = {key: torch.tensor(value) for key, value in encoding.items()}

        if is_label_present:
            encoding['labels'] = torch.tensor(label)
            
        return encoding

    def __len__(self):
        return len(self.json_list)
