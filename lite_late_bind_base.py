import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
from data import TypingDataset
import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm, trange
import time
import random
import json
from transformers import RobertaForSequenceClassification, RobertaConfig
from transformers import AutoTokenizer, AdamW
from base_model_large import ConcatMLP, FullModel


logger = logging.getLogger(__name__)
# pretrained_model = "roberta-large-mnli"
pretrained_model = "roberta-large"

"""
  Model
"""


class roberta_mnli_typing(nn.Module):
    def __init__(self):
        super(roberta_mnli_typing, self).__init__()
        self.roberta_module = RobertaForSequenceClassification.from_pretrained(pretrained_model)
        self.config = RobertaConfig.from_pretrained(pretrained_model)

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta_module(input_ids, attention_mask)
        res = nn.functional.softmax(roberta_output.logits, dim=-1)
        return res


def train(args, train_dataset, model, tokenizer):
    curr_time = time.strftime("%H_%M_%S_%b_%d_%Y", time.localtime())
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=lambda x: zip(*x))

    # set up optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    margin_criterion = torch.nn.MarginRankingLoss(margin=args.margin).to(args.device)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Start Training
    logger.info("***** Starting training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch Size = %d", args.train_batch_size)

    global_step = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")


    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        loss_stat = []
        for step, batch in enumerate(epoch_iterator):
            model.train()
            premise_lst, entity_lst, pos_lst, pos_general_lst, pos_fine_lst, pos_ultrafine_lst = [list(item) for item in batch]

            dat_prem = []
            dat_hypo_true = []
            dat_hypo_false = []

            depend_prem = []
            depend_hypo_true = []
            depend_hypo_false = []

            for idx in range(len(premise_lst)):
                premise = premise_lst[idx]
                entity = entity_lst[idx]
                label = pos_lst[idx]
                general = pos_general_lst[idx]
                fine = pos_fine_lst[idx]
                ultrafine = pos_ultrafine_lst[idx]

                pos = random.sample(label, 1)[0]
                neg = random.sample([tmp for tmp in train_dataset.label_lst if tmp not in pos_lst], 1)[0]

                pos_input_temp = ' '.join([entity, 'is a', pos+'.'])
                neg_input_temp = ' '.join([entity, 'is a', neg+'.'])

                dat_prem.append(premise)
                dat_hypo_true.append(pos_input_temp)
                dat_hypo_false.append(neg_input_temp)

                # dependency
                if pos in ultrafine:
                    try:
                        pos_father = random.sample(fine + general, 1)[0]
                    except:
                        continue
                elif pos in fine:
                    try:
                        pos_father = random.sample(general, 1)[0]
                    except:
                        continue
                else:  # true label is a general label
                    continue

                # discuss about father
                if pos_father in fine:
                    pos_father_neg = random.sample([tmp for tmp in train_dataset.fine_lst if tmp not in label], 1)[0]
                elif pos_father in general:
                    pos_father_neg = random.sample([tmp for tmp in train_dataset.general_lst if tmp not in label], 1)[0]
                else:
                    continue

                depend_prem_temp = ' '.join([entity, 'is a', pos + '.'])
                depend_pos_input_temp = ' '.join([entity, 'is a', pos_father + '.'])
                depend_neg_input_temp = ' '.join([entity, 'is a', pos_father_neg + '.'])

                depend_prem.append(depend_prem_temp)
                depend_hypo_true.append(depend_pos_input_temp)
                depend_hypo_false.append(depend_neg_input_temp)

            indicator = torch.tensor(np.ones(len(dat_prem), dtype=np.float32), requires_grad=False).to(args.device)

            # true
            # print(tokenizer(dat_prem, padding=True, return_tensors='pt').values())
            input_ids_dat_prem, attn_mask_dat_prem = tokenizer(dat_prem, padding=True, return_tensors='pt').values()
            input_ids_dat_hypo_true, attn_mask_dat_hypo_true = tokenizer(dat_hypo_true, padding=True, return_tensors='pt').values()
            # TODO: 检查autotokenizer那个地方，不知道是不是这样用的
            # print("input_ids_dat_prem: ", input_ids_dat_prem)
            # print("attn_mask_dat_prem: ", attn_mask_dat_prem)

            input_ids_dat_prem = input_ids_dat_prem.to(args.device)
            attn_mask_dat_prem = attn_mask_dat_prem.to(args.device)
            input_ids_dat_hypo_true = input_ids_dat_hypo_true.to(args.device)
            attn_mask_dat_hypo_true = attn_mask_dat_hypo_true.to(args.device)

            output = model(input_ids_dat_prem, input_ids_dat_hypo_true, attn_mask_dat_prem, attn_mask_dat_hypo_true)[:, -1]

            # false
            input_ids_dat_hypo_false, attn_mask_dat_hypo_false = tokenizer(dat_hypo_false, padding=True, return_tensors='pt').values()

            input_ids_dat_hypo_false = input_ids_dat_hypo_false.to(args.device)
            attn_mask_dat_hypo_false = attn_mask_dat_hypo_false.to(args.device)

            output_false = model(input_ids_dat_prem, input_ids_dat_hypo_false, attn_mask_dat_prem, attn_mask_dat_hypo_false)[:, -1]

            loss = margin_criterion(output, output_false, indicator)
            indicator = None

            if depend_hypo_true:
                indicator = torch.tensor(np.ones(len(depend_hypo_true), dtype=np.float32),
                                         requires_grad=False).to(args.device)
                # true
                input_ids_depend_prem, attn_mask_depend_prem = tokenizer(depend_prem, padding=True, return_tensors='pt').values()
                input_ids_depend_hypo_true, attn_mask_depend_hypo_true = tokenizer(depend_hypo_true, padding=True, return_tensors='pt').values()

                input_ids_depend_prem = input_ids_depend_prem.to(args.device)
                attn_mask_depend_prem = attn_mask_depend_prem.to(args.device)
                input_ids_depend_hypo_true = input_ids_depend_hypo_true.to(args.device)
                attn_mask_depend_hypo_true = attn_mask_depend_hypo_true.to(args.device)

                output_depend = model(input_ids_depend_prem, input_ids_depend_hypo_true, attn_mask_depend_prem, attn_mask_depend_hypo_true)[:, -1]

                # false
                input_ids_depend_hypo_false, attn_mask_depend_hypo_false = tokenizer(depend_hypo_false, padding=True, return_tensors='pt').values()
                
                input_ids_depend_hypo_false = input_ids_depend_hypo_false.to(args.device)
                attn_mask_depend_hypo_false = attn_mask_depend_hypo_false.to(args.device)

                output_depend_false = model(input_ids_depend_prem, input_ids_depend_hypo_false, attn_mask_depend_prem, attn_mask_depend_hypo_false)[:, -1]

                loss_depend = margin_criterion(output_depend, output_depend_false, indicator)

                loss += args.lamb * loss_depend

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_stat.append(loss.data.cpu().numpy())

        global_step += 1
        logging.info(f'finished with loss ={np.average(loss_stat)}\n')

        if global_step > 0 and global_step % args.save_epochs == 0:
            # Note: Modified for continual learning on FIGER
            # training_details = f'late_bind_MLP_epochs{global_step}_batch{args.train_batch_size}_margin{args.margin}' \
            #                    f'_lr{args.learning_rate}lambda{args.lamb}_{curr_time}'
            training_details = f'late_bind_MLP_figer_epochs{global_step}_batch{args.train_batch_size}_margin{args.margin}' \
                               f'_lr{args.learning_rate}lambda{args.lamb}_{curr_time}'
            MODEL_SAVING_PATH = os.path.join(args.output_dir, training_details)
            saving_checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(saving_checkpoint, MODEL_SAVING_PATH)
            logging.info(f"***Saved model to {MODEL_SAVING_PATH}***\n")




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        type=str,
                        default='/nas/home/mingtaod/codes/lite/data',
                        help="The input data directory.")
    parser.add_argument("--output_dir",
                        type=str,
                        default='/nas/home/mingtaod/codes/lite/output/616_ontonotes_tuned_lite',
                        help="The output directory where the model will be saved.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        # default=4,
                        type=int,
                        help="Total batch size for training.")

    # training arguments
    parser.add_argument("--learning_rate",
                        default=1e-6,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=2600,
                        # Note: FIGER modified
                        # default=100,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--margin",
                        default=0.1,
                        type=float,
                        help="Margin for the margin ranking loss")
    parser.add_argument("--save_epochs",
                        default=50,
                        # Note: FIGER modified
                        # default=1,
                        type=int,
                        help="Save checkpoint every X epochs of training")
    parser.add_argument("--lamb",
                        default=0.05,
                        type=float,
                        help="Margin for the margin ranking loss")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="Weight deay of the optimizer.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        # default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.data_dir):
        raise ValueError("Cannot find data_dir: {}".format(args.gradient_accumulation_steps))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, which should be >= 1".format(
            args.gradient_accumulation_steps))

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")  # To change the device, modify here
    args.device = device

    # Setup logging
    cur_time = time.strftime("%H_%M_%S_%b_%d_%Y", time.localtime())
    logging.basicConfig(filename=os.path.join(args.output_dir, 
                        f'logs_late_bind_MLP_batch{args.train_batch_size}_margin{args.margin}' \
                        f'_lr{args.learning_rate}lambda{args.lamb}_{cur_time}.log'),
                        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # model
    linear_dim = 3
    batch_size = 2

    model = FullModel(linear_dim=linear_dim, batch_size=batch_size).to(device)
    # checkpoint = torch.load('pretrained_weights/weights_base_large_lr1/model_weights_epoch_27.ckpt', map_location='cuda:0')
    # Continual learning
    checkpoint = torch.load('output/late_bind_MLP_figer_lr1_finetune_stepwise/late_bind_MLP_figer_step320000_18_52_07_Jun_03_2022', map_location='cuda:5')
    model.load_state_dict(checkpoint['model'])

    model.to(device)
    logging.info(f'###\nModel Loaded to {torch.cuda.get_device_name(device)}, cuda:5')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    # training data
    train_dataset = TypingDataset(os.path.join(args.data_dir, "train_processed.json"), os.path.join(args.data_dir, "compiled_types.txt"))
    # Note: FIGER modified
    # train_dataset = TypingDataset(os.path.join(args.data_dir, "full_processed.json"), os.path.join(args.data_dir, "types.txt"))
    # train_dataset = TypingDataset(os.path.join(args.data_dir, "g_full_tree_processed.json"), os.path.join(args.data_dir, "compiled_types.txt"))

    # train
    train(args, train_dataset, model, tokenizer)


if __name__ == "__main__":
    main()
