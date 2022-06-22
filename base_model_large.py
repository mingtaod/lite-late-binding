import os
import shutil
from statistics import mode
from turtle import forward
from pyparsing import ParseElementEnhance, original_text_for
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerBase
from datasets import load_dataset, load_metric
import time

from main_roberta_large import PLMEncoder

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

# One time usage
tot_time_MLP = 0
count_num_MLP = 0


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ConcatMLP(nn.Module):
    def __init__(self, linear_dim, batch_size) -> None:
        super(ConcatMLP, self).__init__()

        self.linear_dimension = linear_dim
        self.batch_size = batch_size
        
        self.linear_module = nn.Sequential(
            # Complex core version
            # nn.Linear(1024*5, 8192),
            # nn.GELU(),

            # nn.Linear(8192, 4096), 
            # nn.GELU(),

            # nn.Linear(4096, 2048), 
            # nn.GELU(),

            # nn.Linear(2048, 1024), 
            # nn.GELU(),

            # nn.Linear(1024, 512),
            # nn.GELU(),

            # nn.Linear(512, 256), 
            # nn.GELU(),

            # nn.Linear(256, 64), 
            # nn.GELU(),

            # nn.Linear(64, linear_dim),
            # nn.GELU()

            # Basic core version
            nn.Linear(1024*5, 4096),
            nn.GELU(),

            nn.Linear(4096, 2048), 
            nn.GELU(),

            nn.Linear(2048, 1024), 
            nn.GELU(),

            nn.Linear(1024, 256),
            nn.GELU(),

            nn.Linear(256, linear_dim),
            nn.GELU()
        )

        # Softmax layer
        self.softmax_layer = nn.Softmax(dim=1)


    def forward(self, rep1, rep2):
        time_start=time.time()
        # Defining pooling operations here
        # max_pool = torch.unsqueeze(torch.maximum(rep1, rep2), 0)
        max_pool = torch.maximum(rep1, rep2)
        abs_pool = torch.abs(rep1 - rep2)
        mult_pool = rep1.mul(rep2)  # dim->(N, E)

        # Concatenating the multiple pooling results together
        pool_concat = torch.cat((max_pool, rep1, abs_pool, rep2, mult_pool), 1)  # dim->(N, 1024*5)

        # TODO: 这个如果直接调用训练好的模型权重的话，好像没有任何用处，所以先注释掉
        # orig_dim = pool_concat.shape[0]
        # if orig_dim != self.batch_size:
        #     diff = self.batch_size - orig_dim
        #     torch_tensor = torch.zeros([diff, pool_concat.shape[1]]).to(device)
        #     pool_concat = torch.cat((pool_concat, torch_tensor), 0)

        out = self.linear_module(pool_concat)
        # out = out[:orig_dim, :]
        out = self.softmax_layer(out)
        time_end=time.time()

        global count_num_MLP
        global tot_time_MLP

        if count_num_MLP < 12:
            tot_time_MLP += (time_end-time_start)
            count_num_MLP += 1
            if count_num_MLP == 12:
                print("tot_time_MLP = ", tot_time_MLP)

        return out


class FullModel(nn.Module):
    def __init__(self, linear_dim, batch_size) -> None:
        super(FullModel, self).__init__()
        self.rep_module = PLMEncoder()
        self.mlp_module = ConcatMLP(linear_dim, batch_size)

    def forward(self, input_id_1, input_id_2, attention_mask_1, attention_mask_2):
        sentence_rep_1 = torch.squeeze(self.rep_module(input_id_1, attention_mask_1)[:, 0, :], 1)
        sentence_rep_2 = torch.squeeze(self.rep_module(input_id_2, attention_mask_2)[:, 0, :], 1)  # dim=(N, E), E=embedding size
        concat_mlp_out = self.mlp_module(sentence_rep_1, sentence_rep_2)
        return concat_mlp_out
        

def train(mnli_dataloader, in_model, in_optimizer, curr_epoch, in_lists, criterion, plm_metric):
    losses = AverageMeter('Loss', ':.4e')
    correct_num = 0
    total_num = 0
    accum_step_size = 32

    model.train()

    for i, batch in enumerate(mnli_dataloader):
        batch["premise_ids"] = batch["premise_ids"].to(device)
        batch["hypo_ids"] = batch["hypo_ids"].to(device)
        batch["attn_mask_premise"] = batch["attn_mask_premise"].to(device)
        batch["attn_mask_hypo"] = batch["attn_mask_hypo"].to(device)

        outputs = in_model(batch["premise_ids"], batch["hypo_ids"], batch["attn_mask_premise"], batch["attn_mask_hypo"])

        labels = batch["labels"].to(device)
        loss = criterion(outputs, labels)

        # Debug here: 如果加上grad accmu step，不知道这里需不需要改动
        # -> 不需要改动，因为我们需要记录的是真实的loss而不是gradient accumulation中估计出来的loss
        losses.update(loss.item(), n=batch["premise_ids"].size(0))  

        # 这里loss是average over original batch_size的，然后我们再除以accumu step size, 然后把几个step的average相加，
        # 这样大致相当于把整体的loss加起来除以original batch_size * accumu_step_size
        loss = loss / accum_step_size  
    
        predictions = torch.argmax(outputs, 1)
        correct_num += (predictions == labels).sum().float()
        total_num += len(labels)

        loss.backward()

        if ((i + 1) % accum_step_size == 0) or (i + 1 == len(mnli_dataloader)):
            in_optimizer.step()
            in_optimizer.zero_grad()

    accuracy = (correct_num/total_num).cpu().detach()
    in_lists['loss_train'].append(losses.avg)
    in_lists['acc_train'].append(accuracy)
    print(curr_epoch, '-th epoch     ', 'train loss sum: ', losses.sum, '   train loss avg: ', losses.avg, '   train accuracy: ', accuracy)
    return losses.avg, accuracy


def validate(mnli_val_loader, in_model, in_lists, criterion, whether_plot, plm_metric):
    losses = AverageMeter('Loss', ':.4e')
    correct_num = 0
    total_num = 0

    in_model.eval()

    with torch.no_grad():
        for batch in mnli_val_loader:
            batch["premise_ids"] = batch["premise_ids"].to(device)
            batch["hypo_ids"] = batch["hypo_ids"].to(device)
            batch["attn_mask_premise"] = batch["attn_mask_premise"].to(device)
            batch["attn_mask_hypo"] = batch["attn_mask_hypo"].to(device)

            outputs = in_model(batch["premise_ids"], batch["hypo_ids"], batch["attn_mask_premise"], batch["attn_mask_hypo"])

            labels = batch["labels"].to(device)
            loss = criterion(outputs, labels)
            predictions = torch.argmax(outputs, 1)
            correct_num += (predictions == labels).sum().float()
            total_num += len(labels)

            losses.update(loss.item(), n=batch["premise_ids"].size(0))  # Debug here

    accuracy = (correct_num/total_num).cpu().detach()
    in_lists['loss_val'].append(losses.avg)
    in_lists['acc_val'].append(accuracy)
    print('                ', 'valid loss sum: ', losses.sum, '   valid loss avg: ', losses.avg, '   valid accuracy: ', accuracy)
    return losses.avg, accuracy


# TODO: 根据需要改成基于n个iteration进行plot，而不是epochs
def plot_losses(lst_loss, title):
    plt.plot(lst_loss, '-r', label='loss')
    plt.xlabel('nth epoch')
    plt.legend(loc='upper left')
    plt.title(title)
    save_path = os.path.normpath("%s/%s" % ('plots_dir/base_plots_large_lr1_complex', title+'.png'))
    plt.savefig(save_path)
    plt.close()


def plot_accuracies(lst_acc, title):
    plt.plot(lst_acc, '-r', label='accuracy')
    plt.xlabel('nth epoch')
    plt.legend(loc='upper left')
    plt.title(title)
    save_path = os.path.normpath("%s/%s" % ('plots_dir/base_plots_large_lr1_complex', title+'.png'))
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    # Cleaning the weight files in the directory
    if os.path.isdir("weights_dir/weights_base_large_lr1_complex"):
        shutil.rmtree("weights_dir/weights_base_large_lr1_complex")
    os.mkdir("weights_dir/weights_base_large_lr1_complex")
    # 把文件夹名字当作参数传进来

    print("Device: ", device)

    # Model hyper-parameters
    embed_dim = 1024
    num_heads = 1
    linear_dim = 3
    cls_dim = 1024

    # Training parameters
    batch_size = 2
    epochs = 100  # modify this

    # Define training tools
    # model = CombinedModel(embed_dim, num_heads, linear_dim, cls_dim, batch_size).to(device)
    model = FullModel(linear_dim=linear_dim, batch_size=batch_size).to(device)

    # Wrapping the model in a parallel wrapper
    crit = nn.CrossEntropyLoss()

    # Add L2 penalty to the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-6, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, amsgrad=False)

    # Loading dataset & metric & tokenizer from huggingface
    mnli_data = load_dataset("glue", "mnli", download_mode="reuse_dataset_if_exists")
    mnli_metric = load_metric('glue', "mnli",)
    # print(mnli_metric)
    mnli_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    # Defining tokenizer functions
    def tokenize_function(example):
        return mnli_tokenizer(example["premise"], example["hypothesis"], truncation=True)

    def tokenize_function_1(example):
        return mnli_tokenizer(example["premise"], max_length=512, truncation=True, padding="max_length")

    def tokenize_function_2(example):
        return mnli_tokenizer(example["hypothesis"], max_length=512, truncation=True, padding="max_length")

    tokenized_data = mnli_data.map(tokenize_function_1, batched=True)
    tokenized_data = tokenized_data.rename_column(original_column_name="input_ids", new_column_name="premise_ids")
    tokenized_data = tokenized_data.rename_column(original_column_name="attention_mask", new_column_name="attn_mask_premise")

    tokenized_data = tokenized_data.map(tokenize_function_2, batched=True)
    tokenized_data = tokenized_data.rename_column(original_column_name="input_ids", new_column_name="hypo_ids")
    tokenized_data = tokenized_data.rename_column(original_column_name="attention_mask", new_column_name="attn_mask_hypo")
    tokenized_data = tokenized_data.rename_column(original_column_name="label", new_column_name="labels")

    tokenized_data = tokenized_data.remove_columns(["hypothesis", "premise", "idx"])
    tokenized_data.set_format("torch")

    # TODO: to be commented out after successful try
    # Getting a subset of train set for code testing purpose
    # subset_size_train = 30000
    # subset_size_val = 16

    # Uncomment when running on the full set of data
    # subset_train, _ = torch.utils.data.random_split(tokenized_data["train"], [subset_size_train, len(tokenized_data["train"])-subset_size_train])
    # subset_val, _ = torch.utils.data.random_split(tokenized_data["validation_matched"], [subset_size_val, len(tokenized_data["validation_matched"])-subset_size_val])

    # Creating dataloaders for the flow
    train_dataloader = DataLoader(tokenized_data["train"], shuffle=True, batch_size=batch_size)
    # train_dataloader = DataLoader(subset_train, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(tokenized_data["validation_matched"], batch_size=batch_size)
    # val_dataloader = DataLoader(subset_val, batch_size=batch_size)
    test_dataloader = DataLoader(tokenized_data["test_matched"], batch_size=batch_size)

    lst_loss_train = []
    lst_acc_train = []
    lst_loss_val = []
    lst_acc_val = []

    lowest_avg_loss_train = float('inf')
    lowest_avg_loss_val = float('inf')
    highest_acc_train = 0.0
    highest_acc_val = 0.0

    lists = {'loss_train': lst_loss_train,
             'acc_train': lst_acc_train,
             'loss_val': lst_loss_val,
             'acc_val': lst_acc_val
             }

    for epoch in range(0, epochs):
        curr_loss_train, curr_acc_train = train(train_dataloader, model, optimizer, epoch, lists, crit, mnli_metric)
        lowest_avg_loss_train = min(curr_loss_train, lowest_avg_loss_train)
        highest_acc_train = max(curr_acc_train, highest_acc_train)

        print("Currently the lowest avg training loss =", lowest_avg_loss_train,
              "\n          the highest training acc for a epoch =", highest_acc_train)

        curr_loss_val, curr_acc_val = validate(val_dataloader, model, lists, crit, True, mnli_metric)
        lowest_avg_loss_val = min(curr_loss_val, lowest_avg_loss_val)
        highest_acc_val = max(curr_acc_val, highest_acc_val)

        if highest_acc_val == curr_acc_val:
            # Save the model weights when we find the current highest validation accuracy
            torch.save(model.state_dict(), "weights_dir/weights_base_large_lr1_complex/model_weights_epoch_"+str(epoch)+".ckpt")

        print("Currently the lowest avg validation loss =", lowest_avg_loss_val,
              "\n          the highest validation acc for a epoch =", highest_acc_val)
        print("\n")

    plot_losses(lists['loss_train'], 'train_loss_plot')
    plot_losses(lists['loss_val'], 'valid_loss_plot')
    plot_accuracies(lists['acc_train'], 'train_acc_plot')
    plot_accuracies(lists['acc_val'], 'valid_acc_plot')




# TODO: 
# 检查补零的问题