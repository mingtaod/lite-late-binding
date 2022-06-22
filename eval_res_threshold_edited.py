import json
import numpy as np
import os
# from eval_metric import macro, mrr
# from eval_metric import f1
from heapq import nlargest
from sklearn import metrics
import sklearn


def f1(p, r):
    if r == 0.:
        return 0.
    return 2 * p * r / float(p + r)


def merge_dict(dict1, dict2):
    res = {**dict1, **dict2}
    return res

#  loose macro
def macro(threshold,eval_data):
        p = 0.
        pred_example_count = 0
        r = 0.
        gold_label_count = 0
        print(f'Threshold = {threshold}')
        prec = 0
        rec = 0
        for raw_dat in eval_data:
            true_labels = raw_dat['annotation']
            false_labels = raw_dat['false_annotation']
            pred = merge_dict(true_labels, false_labels)
            predicted_labels = [labels for labels in pred if pred[labels] >= threshold]

            true_labels = set(true_labels).intersection(set(label_lst))
            predicted_labels = set(predicted_labels).intersection(set(label_lst))

            if predicted_labels:
                per_p = len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
                pred_example_count += 1
                p += per_p
            if true_labels:
                per_r = len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
                gold_label_count += 1
                r += per_r

        if pred_example_count > 0:
            prec = p / pred_example_count
        if gold_label_count > 0:
            rec = r / gold_label_count
        
        return prec, rec

# Loose-macro follow ultra-fine grained entity typing
# print('Eval on Loose Macro Score:')
# threshold = TRESHOLD_START
# f1_champ = 0
# threshold_champ = threshold
# while threshold < 1:
#     precision, recall = macro(threshold,dev_dat)
#     summary = f'{round(precision, 4) * 100}\t' \
#                 f'{round(recall, 4) * 100}\t' \
#                 f'{round(f1(precision, recall), 4) * 100}'
#     print(summary)

#     if f1(precision, recall) > f1_champ:
#         threshold_champ = threshold
#         f1_champ = f1(precision, recall)
    
#     threshold += STEP


# Now we got threshold champ
# print('\n###\tNow on Test Set\t###\n')
# precision, recall = macro(threshold_champ,test_dat)
# summary = f'{round(precision, 4) * 100}\t' \
#             f'{round(recall, 4) * 100}\t' \
#             f'{round(f1(precision, recall), 4) * 100}'
# print(summary)


if __name__ == '__main__':
    TRESHOLD_START = 0.7
    STEP = 5e-3

    # TAG_FILE_PATH = 'data/types.txt'
    TAG_FILE_PATH = 'figer_ontonotes_data/compiled_types.txt'

    # OUTPUT_DIR = ""

    # output_files = os.listdir(OUTPUT_DIR)
    # TEST_RES_FILE_PATH = [items for items in output_files if 'test' in items]
    # DEV_RES_FILE_PATH = [items for items in output_files if 'dev' in items]

    # 暂时把两个file path的名字互换位置
    # DEV_RES_FILE_PATH = []
    # TEST_RES_FILE_PATH = ['/nas/home/mingtaod/codes/lite/output/late_bind_MLP_figon_lr1_finetune_stepwise/late_bind_MLP_figon_step360000_18_53_43_Jun_03_2022_dev.json']
    # TEST_RES_FILE_PATH = ['output/late_bind_MLP_ontonotes_lr1_finetune_stepwise/late_bind_MLP_ontonotes_step270000_18_41_18_Jun_03_2022_g_dev_tree_processed.json']
    # TEST_RES_FILE_PATH = ['output/late_bind_MLP_ontonotes_lr1_finetune_stepwise/late_bind_MLP_ontonotes_step360000_18_41_18_Jun_03_2022_g_dev_tree_processed.json']
    # TEST_RES_FILE_PATH = ['output/late_bind_MLP_figer_lr1_finetune_stepwise/late_bind_MLP_figer_step160000_18_52_07_Jun_03_2022_dev_processed.json']
    TEST_RES_FILE_PATH = ['output/late_bind_MLP_figer_lr1_finetune_stepwise/late_bind_MLP_figer_step320000_18_52_07_Jun_03_2022_dev_processed.json']

    # Load tag data
    label_lst = []
    with open(TAG_FILE_PATH) as fin:
        for lines in fin:
            lines = lines.split()[0]
            lines = ' '.join(lines.split('_'))
            label_lst.append(lines)
    general_lst = label_lst[0:9]
    fine_lst = label_lst[9:130]
    ultrafine_lst = label_lst[130:]


    # dev_dat = []
    # for PATH in DEV_RES_FILE_PATH:
    #     with open(PATH) as fin:
    #         for lines in fin:
    #             dev_dat.append(json.loads(lines))

    test_dat = []
    for PATH in TEST_RES_FILE_PATH:
        with open(PATH) as fin:
            for lines in fin:
                test_dat.append(json.loads(lines))

    # Edited code: 
    p_s = []
    r_s = []
    thresh = 0
    best_thresh = thresh
    best_f1 = 0
    while thresh < 1:
        precision, recall = macro(thresh,test_dat)
        curr_f1 = f1(precision, recall)

        p_s.append(precision)
        r_s.append(recall)

        summary = f'{round(precision, 4) * 100}\t' \
                    f'{round(recall, 4) * 100}\t' \
                    f'{round(f1(precision, recall), 4) * 100}'
        
        if curr_f1 > best_f1:
            best_f1 = curr_f1
            best_thresh = thresh

        print("Current threshold: ", thresh)
        print(summary)
        print("\n")

        thresh += STEP

    print("Best threshold: ", best_thresh)
    print("Best F1 score: ", best_f1)

    # p_s = np.array(p_s)
    # r_s = np.array(r_s)
    # auc_prc = metrics.auc(p_s, r_s)
    # print("AUC for PR: ", auc_prc)

        