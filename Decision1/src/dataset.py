import torch
from torch.utils.data import Dataset
import json

class Decision1Dataset(Dataset):
    def __init__(
            self,
            json_file,
            tokenizer,
            max_length,
            return_token_type_ids=None
            ):

        decision1dataset = list()
        with open(json_file) as f:
            for l in f:
                d = json.loads(l)
                inst = self.mkDecision1Dataset(d)
                decision1dataset.append(inst)

        self.decision1dataset = decision1dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_token_type_ids=return_token_type_ids

    def mkDecision1Dataset(self, d):
        label = 'negative' if d['note'][0] is None else 'positive'
        inst = {
                'appNum': d['appNum'],
                'paraNum': d['paraNum'],
                'paraTxt': d['paraTxt'],
                'label': label,
                }
        return inst

    def getJsonList(self):
        return self.decision1dataset

    def __getitem__(self, idx):
        d = self.decision1dataset[idx]

        text = d['paraTxt']

        is_label_present = 'label' in d
        if is_label_present:
            label = 1 if d['label'] == 'positive' else 0
        
        encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_token_type_ids=self.return_token_type_ids
                )

        # This makes training unstable for Decision (i) models.
        #encoding['position_ids'] = list(range(self.max_length))

        encoding = {key: torch.tensor(value) for key, value in encoding.items()}

        if is_label_present:
            encoding['labels'] = torch.tensor(label)

        return encoding

    def __len__(self):
        return len(self.decision1dataset)
