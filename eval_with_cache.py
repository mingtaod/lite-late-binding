import os
# 这一行暂时注释掉
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
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
from eval import TAG_FILE_PATH

parser = argparse.ArgumentParser(description='eval_batch')
parser.add_argument('--path', type=str, default='optimizerAdamW_batch16_margin0.1_lr5e-06_15_27_30_Sep_09_2021/optimizerAdamW_epochs1650_batch16_margin0.1_lr5e-06_15_27_30_Sep_09_2021', nargs='?')
parser.add_argument('--test', type=str, default='data/test_processed.json', nargs='?')
parser.add_argument('--batch', type=int, default=8, nargs='?')
parser.add_argument('--check', type=str, default='roberta-large-mnli', nargs='?')
args = parser.parse_args()

DEVICE = 0
EVAL_BATCH = args.batch
THRESH = 0.6

# ---> To switch the types file: edit hefre
# TAG_FILE_PATH = 'data/ufet_types.txt'
# TAG_FILE_PATH = 'figer_ontonotes_data/figon_types.txt'
TAG_FILE_PATH = 'ontonotes_data/processed/ontonotes_types.txt'
# TAG_FILE_PATH = 'data/compiled_types.txt'

MODEL_PATH = args.path
TEST_SET_PATH = args.test

EVAL_SET = TEST_SET_PATH.split('/')[-1]

checkpoint = args.check
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

if checkpoint == "roberta-large-mnli":
    model = FullModel(linear_dim=3, batch_size=2).to(device=DEVICE)
else:
    exit(1)

if __name__ == "__main__":
    # Modified here 6/22
    # First load the cached file
    # --> Switch cache dict here: 
    cache_dict = torch.load('cached_emb/late_bind_MLP_ufet_step150000_18_40_59_Jun_30_2022.pth')
    # cache_dict = torch.load('cached_emb/late_bind_MLP_figon_step1080000_03_28_06_Jun_25_2022.pth')

    chkpt = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(chkpt['model'])
    model.eval()

    # Load tag data
    label_lst = []

    with open(TAG_FILE_PATH) as fin:
        for lines in fin:
            lines = lines.strip('\n')
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

        # Constructing the embedding of the premise
        prem_inputs = tokenizer(premise, padding=True, return_tensors='pt').to(device=DEVICE)
        prem_emb = torch.squeeze(model.rep_module(**prem_inputs)[:, 0, :], 1)

        true_buffer = {}
        for true_batch_id in range(0, len(annotations), EVAL_BATCH):
            dat_buffer = annotations[true_batch_id:true_batch_id+EVAL_BATCH]
            input_buffer_hypo = []

            for annotation in dat_buffer:
                true_sequence_hypo = ' '.join(['The mentioned entity is a', annotation+'.'])
                input_buffer_hypo.append(annotation)

            hypo_emb = torch.Tensor([])
            ind = 0
            for raw_hypo_annot in input_buffer_hypo:
                if ind == 0:
                    hypo_emb = torch.unsqueeze(torch.from_numpy(cache_dict[raw_hypo_annot]), 0)
                else:
                    hypo_emb = torch.cat((hypo_emb, torch.unsqueeze(torch.from_numpy(cache_dict[raw_hypo_annot]), 0)), 0)
                ind += 1

            hypo_emb = hypo_emb.to(device=DEVICE)
            
            prem_emb_1 = prem_emb.repeat_interleave(ind, dim=0)
            true_output = model.mlp_module(prem_emb_1, hypo_emb)[:, -1]

            true_res = true_output.detach().cpu().numpy().tolist()
            for idx in range(len(dat_buffer)):
                true_buffer[dat_buffer[idx]] = true_res[idx]

        res['annotation'] = true_buffer

        # false sampling all
        false_dat = [tmp for tmp in label_lst if tmp not in annotations]
        # print('false_dat: ', false_dat)
        # set bar for logging false samplings
        true_values = list(true_buffer.values())
        bar = np.min(true_values)

        false_buffer = {}
        for false_batch_id in range(0,len(false_dat),EVAL_BATCH):
            dat_buffer = false_dat[false_batch_id:false_batch_id+EVAL_BATCH]

            dat_false_prem = []
            dat_false_hypo = []

            for false_label in dat_buffer:
                dat_false_hypo.append(false_label)

            # false_prem_inputs = tokenizer(premise, padding=True, return_tensors='pt').to(device=DEVICE)

            false_hypo_emb = []
            ind = 0
            for raw_false_label in dat_false_hypo:
                if ind == 0:
                    false_hypo_emb = torch.unsqueeze(torch.from_numpy(cache_dict[raw_false_label]), 0)
                else:
                    false_hypo_emb = torch.cat((false_hypo_emb, torch.unsqueeze(torch.from_numpy(cache_dict[raw_false_label]), 0)), 0)
                ind += 1
            
            false_hypo_emb = false_hypo_emb.to(device=DEVICE)

            prem_emb_2 = prem_emb.repeat_interleave(ind, dim=0)
            false_output = model.mlp_module(prem_emb_2, false_hypo_emb)[:, -1]

            false_res = false_output.detach().cpu().numpy().tolist()
            for idx in range(len(dat_buffer)):
                false_buffer[dat_buffer[idx]] = false_res[idx]

        # --> Switch the lines below for modes saving space or not saving space
        false_annotations = {labels: false_buffer[labels] for labels in false_buffer
                            if (false_buffer[labels] > bar or false_buffer[labels] > THRESH)}
        # false_annotations = {labels: false_buffer[labels] for labels in false_buffer}

        res['false_annotation'] = false_annotations

        # save
        true_inputs = None
        false_inputs = None
        stat_dict.append(res)

    # save res file
    with open(RES_SAVING_PATH, 'w+') as fout:
        fout.write("\n".join([json.dumps(items) for items in stat_dict]))
