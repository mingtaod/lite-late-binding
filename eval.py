import os
# 这一行暂时注释掉
# os.environ["CUDA_VISIBLE_DEVICES"]="7"
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time
import random
import json
from heapq import nlargest
from transformers import RobertaForSequenceClassification, RobertaConfig
from transformers import AutoTokenizer
import argparse
from base_model_large import ConcatMLP, FullModel

parser = argparse.ArgumentParser(description='eval_batch')
parser.add_argument('--path', type=str, default='optimizerAdamW_batch16_margin0.1_lr5e-06_15_27_30_Sep_09_2021/optimizerAdamW_epochs1650_batch16_margin0.1_lr5e-06_15_27_30_Sep_09_2021', nargs='?')
parser.add_argument('--test', type=str, default='data/test_processed.json', nargs='?')
parser.add_argument('--batch', type=int, default=8, nargs='?')
parser.add_argument('--check', type=str, default='roberta-large-mnli', nargs='?')
args = parser.parse_args()

DEVICE = 0
EVAL_BATCH = args.batch
THRESH = 0.6

TAG_FILE_PATH = 'data/types.txt'
MODEL_PATH = args.path
TEST_SET_PATH = args.test

EVAL_SET = TEST_SET_PATH.split('/')[-1]

checkpoint = args.check
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


#  Load model
class roberta_mnli_typing(nn.Module):
    def __init__(self):
        super(roberta_mnli_typing, self).__init__()
        self.roberta_module = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli")
        self.config = RobertaConfig.from_pretrained("roberta-large-mnli")

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta_module(input_ids, attention_mask)
        res = torch.nn.functional.softmax(roberta_output.logits, dim=-1)
        return res

if checkpoint == "roberta-large-mnli":
    # model = roberta_mnli_typing()
    linear_dim = 3
    batch_size = 2
    model = FullModel(linear_dim=linear_dim, batch_size=batch_size).to(device=DEVICE)
else:
    exit(1)

if __name__ == "__main__":
    chkpt = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(chkpt['model'])
    model.to(device=DEVICE)
    model.eval()

    # Load tag data
    label_lst = []
    with open(TAG_FILE_PATH) as fin:
        for lines in fin:
            lines = lines.split()[0]
            lines = ' '.join(lines.split('_'))
            label_lst.append(lines)

    """
    Eval
    """
    RES_SAVING_PATH = f'{MODEL_PATH}_{EVAL_SET}'
    print(f'Evaluate {MODEL_PATH}\n on {TEST_SET_PATH} \n saved to {RES_SAVING_PATH}')

    stat_dict = []

    test_dat = []
    with open(TEST_SET_PATH) as fin:
        for lines in fin:
            test_dat.append(json.loads(lines))

    for raw_dat in tqdm(test_dat):
        res = {}

        premise = raw_dat['premise']
        entity = raw_dat['entity']
        annotations = raw_dat['annotation']

        res['premise'] = premise
        res['entity'] = entity

        true_buffer = {}
        for true_batch_id in range(0, len(annotations), EVAL_BATCH):
            dat_buffer = annotations[true_batch_id:true_batch_id+EVAL_BATCH]
            input_buffer_prem = []
            input_buffer_hypo = []

            for annotation in dat_buffer:
                true_sequence_prem = premise
                true_sequence_hypo = f'{entity} is {annotation}'
                # true_sequence = f'{premise}</s></s>{entity} is {annotation}'
                input_buffer_prem.append(true_sequence_prem)
                input_buffer_hypo.append(true_sequence_hypo)

            # true_inputs = tokenizer(input_buffer, padding=True, return_tensors='pt').to(device=DEVICE)
            prem_ids, prem_mask = tokenizer(input_buffer_prem, padding=True, return_tensors='pt').values()
            hypo_ids, hypo_mask = tokenizer(input_buffer_hypo, padding=True, return_tensors='pt').values()
            prem_ids = prem_ids.to(device=DEVICE)
            prem_mask = prem_mask.to(device=DEVICE)
            hypo_ids = hypo_ids.to(device=DEVICE)
            hypo_mask = hypo_mask.to(device=DEVICE)

            # true_output = model(**true_inputs)[:, -1]
            true_output = model(prem_ids, hypo_ids, prem_mask, hypo_mask)[:, -1]
            true_res = true_output.detach().cpu().numpy().tolist()
            for idx in range(len(dat_buffer)):
                true_buffer[dat_buffer[idx]] = true_res[idx]

        res['annotation'] = true_buffer

        # false sampling all
        false_dat = [tmp for tmp in label_lst if tmp not in annotations]
        # set bar for logging false samplings
        true_values = list(true_buffer.values())
        bar = np.min(true_values)

        false_buffer = {}
        for false_batch_id in range(0,len(false_dat),EVAL_BATCH):
            dat_buffer = false_dat[false_batch_id:false_batch_id+EVAL_BATCH]
            # dat_false = []
            dat_false_prem = []
            dat_false_hypo = []

            for false_label in dat_buffer:
                # false_sequence = f'{premise}</s></s>{entity} is {false_label}'
                false_seq_prem = premise
                false_seq_hypo = f'{entity} is {false_label}'
                # dat_false.append(false_sequence)
                dat_false_prem.append(false_seq_prem)
                dat_false_hypo.append(false_seq_hypo)

            # false_inputs = tokenizer(dat_false, padding=True, return_tensors='pt').to(device=DEVICE)
            # false_output = model(**false_inputs)[:, -1]
            false_prem_ids, false_prem_mask = tokenizer(dat_false_prem, padding=True, return_tensors='pt').values()
            false_hypo_ids, false_hypo_mask = tokenizer(dat_false_hypo, padding=True, return_tensors='pt').values()
            false_prem_ids = false_prem_ids.to(device=DEVICE)
            false_prem_mask = false_prem_mask.to(device=DEVICE)
            false_hypo_ids = false_hypo_ids.to(device=DEVICE)
            false_hypo_mask = false_hypo_mask.to(device=DEVICE)

            false_output = model(false_prem_ids, false_hypo_ids, false_prem_mask, false_hypo_mask)[:, -1]
            false_res = false_output.detach().cpu().numpy().tolist()
            for idx in range(len(dat_buffer)):
                false_buffer[dat_buffer[idx]] = false_res[idx]

        false_annotations = {labels: false_buffer[labels] for labels in false_buffer
                            if (false_buffer[labels] > bar or false_buffer[labels] > THRESH)}

        res['false_annotation'] = false_annotations

        # save
        true_inputs = None
        false_inputs = None
        stat_dict.append(res)

    # save res file
    with open(RES_SAVING_PATH, 'w+') as fout:
        fout.write("\n".join([json.dumps(items) for items in stat_dict]))
