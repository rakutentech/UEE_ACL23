import torch
from torch.utils.data import Dataset
import json

class JSNLIDataset(Dataset):
    def __init__(
            self,
            jsnli_file,
            tokenizer,
            max_length,
            return_token_type_ids=None
            ):

        jsnlidataset = list()
        with open(jsnli_file) as f:
            for l in f:
                l = l.strip()
                label, premise, hypothesis = l.split("\t")
                # Condition B: Ent as Pos, Neu as Neg (ignoring Cont)
                if label == "contradiction": continue
                label = 1 if label == "entailment" else 0
                premise = premise.replace(" ", "")
                hypothesis = hypothesis.replace(" ", "")
                jsnlidataset.append({
                    'label': label,
                    'premise': premise,
                    'hypothesis': hypothesis
                    })

        self.jsnlidataset = jsnlidataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_token_type_ids=return_token_type_ids

    def __getitem__(self, idx):
        d = self.jsnlidataset[idx]
        label = d['label']
        premise = d['premise']
        hypothesis = d['hypothesis']

        encoding = self.tokenizer(
                premise,
                hypothesis,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_token_type_ids=self.return_token_type_ids
                )

        encoding['position_ids'] = list(range(self.max_length))

        encoding = {key: torch.tensor(value) for key, value in encoding.items()}

        encoding['labels'] = torch.tensor(label)

        return encoding

    def __len__(self):
        return len(self.jsnlidataset)

class Decision2Dataset(Dataset):
    def __init__(
            self,
            json_file,
            tokenizer,
            max_length,
            return_token_type_ids=None
            ):

        decision2dataset = list()
        with open(json_file) as f:
            for l in f:
                d = json.loads(l)
                inst1, inst2 = self.mkDecision2Dataset(d)
                if inst1 is None or inst2 is None: continue
                decision2dataset.extend([inst1, inst2])

        self.decision2dataset = decision2dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_token_type_ids=return_token_type_ids

    def mkDecision2Dataset(self, d):
        ueeLabel = d['label']
        note = d['note'][0]

        if ueeLabel == 'positive':
            posClaimsTxt_li = list()
            for cp in d['contClaims']: # contClaims, not claims!
                li = cp['claimTxt']
                posClaimsTxt_li.extend(li)
            posClaimsTxt = ''.join(posClaimsTxt_li)

            negClaimsTxt_li = list()
            for cp in d['claims']: # claims, not contClaims!
                li = cp['claimTxt']
                negClaimsTxt_li.extend(li)
            negClaimsTxt = ''.join(negClaimsTxt_li)

            posInst = {
                    'appNum': d['appNum'],
                    'paraNum': d['paraNum'],
                    'paraTxt': d['paraTxt'],
                    'claimsTxt': posClaimsTxt,
                    'label': 'positive',
                    'decision2Type': 'posFromUEEPos',
                    }

            negInst = {
                    'appNum': d['appNum'],
                    'paraNum': d['paraNum'],
                    'paraTxt': d['paraTxt'],
                    'claimsTxt': negClaimsTxt,
                    'label': 'negative',
                    'decision2Type': 'negFromUEEPos',
                    }

            return posInst, negInst

        elif ueeLabel == 'negative':
            if note is None:
                return None, None
            else:
                posClaimsTxt_li = list()
                for cp in d['claims']: # claims, not contClaims!
                    li = cp['claimTxt']
                    posClaimsTxt_li.extend(li)
                posClaimsTxt = ''.join(posClaimsTxt_li)

                negClaimsTxt_li = list()
                for cp in d['contClaims']: # contClaims, not claims!
                    li = cp['claimTxt']
                    negClaimsTxt_li.extend(li)
                negClaimsTxt = ''.join(negClaimsTxt_li)

                posInst = {
                        'appNum': d['appNum'],
                        'paraNum': d['paraNum'],
                        'paraTxt': d['paraTxt'],
                        'claimsTxt': posClaimsTxt,
                        'label': 'positive',
                        'decision2Type': 'posFromUEENeg',
                        }

                negInst = {
                        'appNum': d['appNum'],
                        'paraNum': d['paraNum'],
                        'paraTxt': d['paraTxt'],
                        'claimsTxt': negClaimsTxt,
                        'label': 'negative',
                        'decision2Type': 'negFromUEENeg',
                        }

                return posInst, negInst

        else:
            raise NotImplementedError

    def getJsonList(self):
        return self.decision2dataset

    def __getitem__(self, idx):
        d = self.decision2dataset[idx]

        text1 = d['paraTxt']
        text2 = d['claimsTxt']

        is_label_present = 'label' in d
        if is_label_present:
            label = 1 if d['label'] == 'positive' else 0
        
        encoding = self.tokenizer(
                text1,
                text2,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_token_type_ids=self.return_token_type_ids
                )

        encoding['position_ids'] = list(range(self.max_length))

        encoding = {key: torch.tensor(value) for key, value in encoding.items()}

        if is_label_present:
            encoding['labels'] = torch.tensor(label)

        return encoding

    def __len__(self):
        return len(self.decision2dataset)
