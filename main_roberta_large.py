import os
import shutil
import torch
import time
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerBase
from datasets import load_dataset, load_metric

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

# One time usage
tot_time_rep = 0
count_num_rep = 0


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


class PLMEncoder(nn.Module):
    def __init__(self):
        super(PLMEncoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained('roberta-large')

    def forward(self, input_ids, attention_mask):
        time_start=time.time()
        out = self.encoder(input_ids, attention_mask)
        time_end=time.time()

        global tot_time_rep
        global count_num_rep
        if count_num_rep < 12:
            tot_time_rep += (time_end-time_start)
            count_num_rep += 1
            if count_num_rep == 12:
                print("tot_time_rep = ", tot_time_rep)

        return out.last_hidden_state


# Debug: 设定为Batch是第二个dimension
# query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
#           the embedding dimension.
class CoreMatchingModel(nn.Module):
    def __init__(self, embed_dim, num_heads, linear_dim, cls_dim, batch_size):
        super(CoreMatchingModel, self).__init__()
        self.batch_size = batch_size

        # Attention layers
        self.cross_attn_cand = nn.MultiheadAttention(embed_dim, num_heads)  
        self.cross_attn_query = nn.MultiheadAttention(embed_dim, num_heads)
        self.self_attn_cand = nn.MultiheadAttention(embed_dim, num_heads)
        self.self_attn_query = nn.MultiheadAttention(embed_dim, num_heads)
        self.attn_pooling = nn.MultiheadAttention(linear_dim, num_heads)

        # Linear dense layers
        self.linear_cand_module = nn.Sequential(
            nn.Linear(embed_dim + cls_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=0.2),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(p=0.2),

            nn.Linear(64, linear_dim),
            nn.BatchNorm1d(linear_dim),
            nn.GELU(),
            # nn.Dropout(p=0.2)
        )

        self.linear_query_module = nn.Sequential(
            nn.Linear(embed_dim + cls_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=0.2),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(p=0.2),

            nn.Linear(64, linear_dim),
            nn.BatchNorm1d(linear_dim),
            nn.GELU(),
            # nn.Dropout(p=0.2)
        )

        # Pooling weight tensor & initialization
        self.combo_vec = nn.Parameter(data=torch.rand(1, batch_size, linear_dim), requires_grad=True)
        nn.init.xavier_normal(self.combo_vec.data)

        # Softmax layer
        self.softmax_layer = nn.Softmax(dim=1)

    # Note: rep_1 is the candidate, while rep_2 is the query
    def forward(self, sentence_rep_1, sentence_rep_2):
        # Computing flow defined in the paper
        sentence_rep_1 = torch.transpose(sentence_rep_1, 0, 1)
        sentence_rep_2 = torch.transpose(sentence_rep_2, 0, 1)

        cand_branch_out, cand_branch_weights = self.cross_attn_cand(sentence_rep_2, sentence_rep_1, sentence_rep_1)
        query_branch_out, query_branch_weights = self.cross_attn_query(sentence_rep_1, sentence_rep_2, sentence_rep_2)

        cand_branch_out, cand_branch_weights = self.self_attn_cand(torch.unsqueeze(cand_branch_out[0, :, :], 0), cand_branch_out, cand_branch_out)
        query_branch_out, query_branch_weights = self.self_attn_query(torch.unsqueeze(query_branch_out[0, :, :], 0), query_branch_out, query_branch_out)

        cand_branch_out = torch.squeeze(cand_branch_out, 0)  # dim->(N, E)
        query_branch_out = torch.squeeze(query_branch_out, 0)

        cand_branch_out = torch.cat((cand_branch_out, sentence_rep_1[0, :, :]), 1)  # dim->(N, E+cls_dim)
        query_branch_out = torch.cat((query_branch_out, sentence_rep_2[0, :, :]), 1)

        # Special treatment code -> given that the batchsize=2, duplicate the sentence rep if the current bach has size 1
        if sentence_rep_1.shape[1] == 1:
            cand_branch_out = torch.cat((cand_branch_out, cand_branch_out), dim=0)
            query_branch_out = torch.cat((query_branch_out, query_branch_out), dim=0)

        cand_branch_out = self.linear_cand_module(cand_branch_out)
        # TODO: check->为什么这里0和1的数值是不一样的？
        # print("\ncand_branch_out[0]: ", cand_branch_out[0])
        # print("\ncand_branch_out[1]: ", cand_branch_out[1])
        query_branch_out = self.linear_query_module(query_branch_out)

        if sentence_rep_1.shape[1] == 1:
            cand_branch_out = torch.unsqueeze(cand_branch_out[0], 0)
            query_branch_out = torch.unsqueeze(query_branch_out[0], 0)

        # Defining pooling operations here
        # dim->(1, N, linear_dim)
        max_pool = torch.unsqueeze(torch.maximum(cand_branch_out, query_branch_out), 0)
        abs_pool = torch.unsqueeze(torch.abs(cand_branch_out - query_branch_out), 0)
        mult_pool = torch.unsqueeze(cand_branch_out.mul(query_branch_out), 0)
        cand_branch_out = torch.unsqueeze(cand_branch_out, 0)  
        query_branch_out = torch.unsqueeze(query_branch_out, 0)

        # Concatenating the multiple pooling results together
        # dim->(5, N, linear_dim)
        pool_concat = torch.cat((max_pool, cand_branch_out, abs_pool, query_branch_out, mult_pool), 0)

        # Attention for pooling w.r.t combo vector
        # Debug: 2/24/2022 -> 这里的补零操作是否合理？感觉这里好像不太对，需要改一下 
        #        -> 思考一下怎么才能不考虑batch里面的元素有多少个也可以无任何问题的将input pass进去
        orig_dim = pool_concat.shape[1]
        if orig_dim != self.batch_size:
            diff = self.batch_size - orig_dim
            torch_tensor = torch.zeros([pool_concat.shape[0], diff, pool_concat.shape[2]]).to(device)
            pool_concat = torch.cat((pool_concat, torch_tensor), 1)

        pool_attn_res, pool_attn_weights = self.attn_pooling(self.combo_vec, pool_concat, pool_concat)
        pool_attn_res = torch.squeeze(pool_attn_res, 0)
        pool_attn_res = pool_attn_res[:orig_dim, :]
        pool_attn_res = self.softmax_layer(pool_attn_res)

        return pool_attn_res


class CombinedModel(nn.Module):
    def __init__(self, embed_dim, num_heads, linear_dim, cls_dim, batch_size):
        super(CombinedModel, self).__init__()
        self.core_matching = CoreMatchingModel(embed_dim, num_heads, linear_dim, cls_dim, batch_size)
        self.PLM_encoder = PLMEncoder()

    def forward(self, input_id_1, input_id_2, attention_mask_1, attention_mask_2):
        # attention_mask here makes sense because we adopted maximum length padding strategy ->
        # later null entries in the padding sequence should not have attention applied on them
        sentence_rep_1 = self.PLM_encoder(input_id_1, attention_mask_1)
        sentence_rep_2 = self.PLM_encoder(input_id_2, attention_mask_2)

        core_match_out = self.core_matching(sentence_rep_1, sentence_rep_2)
        return core_match_out


def train(mnli_dataloader, in_model, in_optimizer, curr_epoch, in_lists, criterion):
    losses = AverageMeter('Loss', ':.4e')
    correct_num = 0
    total_num = 0
    accum_step_size = 16

    model.train()

    for i, batch in enumerate(mnli_dataloader):
        batch["premise_ids"] = batch["premise_ids"].to(device)
        batch["hypo_ids"] = batch["hypo_ids"].to(device)
        batch["attn_mask_premise"] = batch["attn_mask_premise"].to(device)
        batch["attn_mask_hypo"] = batch["attn_mask_hypo"].to(device)

        outputs = in_model(batch["premise_ids"], batch["hypo_ids"], batch["attn_mask_premise"], batch["attn_mask_hypo"])

        labels = batch["labels"].to(device)
        loss = criterion(outputs, labels)

        losses.update(loss.item(), n=batch["premise_ids"].size(0))  

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


def validate(mnli_val_loader, in_model, in_lists, criterion, whether_plot):
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
    save_path = os.path.normpath("%s/%s" % ('plots_dir/roberta_large_plots', title+'.png'))
    plt.savefig(save_path)
    # plt.show()
    plt.close()


def plot_accuracies(lst_acc, title):
    plt.plot(lst_acc, '-r', label='accuracy')
    plt.xlabel('nth epoch')
    plt.legend(loc='upper left')
    plt.title(title)
    save_path = os.path.normpath("%s/%s" % ('plots_dir/roberta_large_plots', title+'.png'))
    plt.savefig(save_path)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    # Cleaning the weight files in the directory
    if os.path.isdir("weights_dir/weights_roberta_large_lr1_complex"):
        shutil.rmtree("weights_dir/weights_roberta_large_lr1_complex")
    os.mkdir("weights_dir/weights_roberta_large_lr1_complex")

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
    model = CombinedModel(embed_dim, num_heads, linear_dim, cls_dim, batch_size).to(device)

    # Wrapping the model in a parallel wrapper
    crit = nn.CrossEntropyLoss()

    # Add L2 penalty to the optimizer
    optimizer = optim.Adam(model.parameters(), lr=5e-6, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, amsgrad=False)

    # Loading dataset & metric & tokenizer from huggingface
    mnli_data = load_dataset("glue", "mnli")
    mnli_metric = load_metric('glue', "mnli")
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
        curr_loss_train, curr_acc_train = train(train_dataloader, model, optimizer, epoch, lists, crit)
        lowest_avg_loss_train = min(curr_loss_train, lowest_avg_loss_train)
        highest_acc_train = max(curr_acc_train, highest_acc_train)

        print("Currently the lowest avg training loss =", lowest_avg_loss_train,
              "\n          the highest training acc for a epoch =", highest_acc_train)

        curr_loss_val, curr_acc_val = validate(val_dataloader, model, lists, crit, whether_plot=True)
        lowest_avg_loss_val = min(curr_loss_val, lowest_avg_loss_val)
        highest_acc_val = max(curr_acc_val, highest_acc_val)

        if highest_acc_val == curr_acc_val:
            # Save the model weights when we find the current highest validation accuracy
            torch.save(model.state_dict(), "weights_dir/weights_roberta_large_lr1_complex/model_weights_epoch_"+str(epoch)+".ckpt")

        print("Currently the lowest avg validation loss =", lowest_avg_loss_val,
              "\n          the highest validation acc for a epoch =", highest_acc_val)
        print("\n")

    plot_losses(lists['loss_train'], 'train_loss_plot')
    plot_losses(lists['loss_val'], 'valid_loss_plot')
    plot_accuracies(lists['acc_train'], 'train_acc_plot')
    plot_accuracies(lists['acc_val'], 'valid_acc_plot')
