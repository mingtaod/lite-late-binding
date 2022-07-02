import os
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, AdamW
from datasets import load_dataset, load_metric
from main_roberta_large import PLMEncoder
from base_model_large import FullModel, ConcatMLP


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
DEVICE = 0

class HypoDataset(Dataset):
    def __init__(self, cands):
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-large')
        input_ids, attn_mask = self.tokenizer(cands, padding=True, return_tensors='pt').values()
        # print('input_ids.shape: ', input_ids.shape)
        # print('attn_mask.shape: ', attn_mask.shape)
        self.cands = cands
        self.input_ids = input_ids
        self.attn_mask = attn_mask
    
    def __getitem__(self, index):
        return {'input_ids':self.input_ids[index], 'attn_mask':self.attn_mask[index]}

    def __len__(self):
        return len(self.cands)

def compute_candidate_emb(type_file_dir, full_model):
    lines = open(type_file_dir).readlines()
    templated_cands = []
    # for line in lines:
    for i in range(len(lines)):
        line = lines[i]
        line = line.strip('\n')
        lines[i] = line
        templated_cand = ' '.join(['The mentioned entity is a', line+'.'])
        templated_cands.append(templated_cand)

    dataset = HypoDataset(templated_cands)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=16)
    epoch_iterator = tqdm(dataloader, desc="Batch")

    full_tensor = np.array([])

    for id, batch in enumerate(epoch_iterator):
        batch['input_ids'] = batch['input_ids'].to(device=DEVICE)
        batch['attn_mask'] = batch['attn_mask'].to(device=DEVICE)
        output_emb = torch.squeeze(full_model.rep_module(batch['input_ids'], batch['attn_mask'])[:, 0, :], 1).cpu().detach().numpy()
        if id == 0: 
            full_tensor = output_emb
        else:
            full_tensor = np.concatenate((full_tensor, output_emb), axis=0)
        # emb_dict[]

    emb_dict = {lines[i]:full_tensor[i] for i in range(len(lines))}
    return emb_dict


if __name__ == '__main__':
    model = FullModel(linear_dim=3, batch_size=2).to(device=DEVICE)
    type_file = 'compiled_types.txt'  # no need to change
    model_dir = 'output/late_bind_MLP_ufet_lr1_finetune_stepwise/late_bind_MLP_ufet_step150000_18_40_59_Jun_30_2022'
    model_id = model_dir.split('/')[2]
    output_dir = os.path.join('cached_emb', model_id+'.pth')

    # Load model first 
    chkpt = torch.load(model_dir, map_location='cpu')
    model.load_state_dict(chkpt['model'])
    model.to(device=DEVICE)
    model.eval()

    # Compute embeddings dictionary given the type file
    embedding_dict = compute_candidate_emb(type_file, model)

    # Saving the dictionary to a file
    torch.save(embedding_dict, output_dir)

    # Test: load
    # dict = torch.load(output_dir)
    # print(dict)