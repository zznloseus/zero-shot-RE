import json
import os
import random
import numpy as np
import time
import torch
import random
import logging
import datetime
import torch.nn as nn
from torch.nn import MSELoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers import get_linear_schedule_with_warmup, AdamW
from argparse import ArgumentParser
from tqdm import tqdm
from utils import *
from dataset import *

import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix


# from model_recall_ir_ot3 import *
# from model_recall_ir_ot3_2 import *

from model_recall_ir_ot3_4 import *
# from model import *
from torch.utils.data import DataLoader, RandomSampler
from tensorboardX import SummaryWriter


def compute_macro_PRF(predicted_idx, gold_idx, i=-1, empty_label=None):
    '''
    This evaluation function follows work from Sorokin and Gurevych(https://www.aclweb.org/anthology/D17-1188.pdf)
    code borrowed from the following link:
    https://github.com/UKPLab/emnlp2017-relation-extraction/blob/master/relation_extraction/evaluation/metrics.py
    '''
    if i == -1:
        i = len(predicted_idx)

    # print(type(predicted_idx))
    # print(predicted_idx)
    # print(type(gold_idx))
    # print(gold_idx)
    complete_rel_set = set(gold_idx) - {empty_label}
    avg_prec = 0.0
    avg_rec = 0.0
    # print(complete_rel_set)
    for r in complete_rel_set:
        # print(i)
        # print(predicted_idx[:i])
        r_indices = (predicted_idx[:i] == r)
        # print(r_indices)
        # print(r_indices.nonzero())
        tp = len((predicted_idx[:i][r_indices] == gold_idx[:i][r_indices]).nonzero()[0])
        tp_fp = len(r_indices.nonzero()[0])
        tp_fn = len((gold_idx == r).nonzero()[0])
        prec = (tp / tp_fp) if tp_fp > 0 else 0
        rec = tp / tp_fn
        avg_prec += prec
        avg_rec += rec
    f1 = 0
    avg_prec = avg_prec / len(set(predicted_idx[:i]))
    avg_rec = avg_rec / len(complete_rel_set)
    if (avg_rec + avg_prec) > 0:
        f1 = 2.0 * avg_prec * avg_rec / (avg_prec + avg_rec)

    return avg_prec, avg_rec, f1

def train(train_dataset, model, args, device):
    writer = SummaryWriter()
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    t_total = len(train_dataloader) * args.epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.1},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warm_up, num_training_steps=t_total)
    
    global_step = 0
    best_step = 0
    min_train_loss = float('inf')
    
    model.zero_grad()
    
    neg_counts = args.epochs
    

    for epoch in range(1, int(args.epochs) + 1):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch, args.epochs))
        print('Training...')
        total_train_loss = 0
        t0 = time.time() 

        for batch in tqdm(train_dataloader, desc="Iteration"):

            model.train()
            inputs = {'sen_input_ids':      batch["input_ids"].to(device),
                    'sen_att_masks':   batch["attention_mask"].to(device),
                    'des_input_ids':      batch["des_input_ids"].to(device),
                    'des_att_masks':   batch["des_attention_mask"].to(device),
                    'marked_e1':        batch["marked_e1"].to(device),
                    'marked_e2':        batch["marked_e2"].to(device),
                    'mark_head':        batch["mark_head"].to(device),
                    'mark_tail':        batch["mark_tail"].to(device),
                    'head_range':        batch["head_range"].to(device),
                    'tail_range':        batch["tail_range"].to(device),
                    'neg_counts':    neg_counts,
                    }

            if inputs['sen_input_ids'].size(0) == 1:
                continue

            outputs = model(**inputs)
            loss = outputs

            total_train_loss += loss.item()
            loss.backward() 

            # print("Parameter: classify.mlp_feature.weight:", f"Gradient: {model.classify.mlp_feature.weight.grad.shape}")
            # print("Parameter: classify.mlp_1.weight:", f"Gradient: {model.classify.mlp_1.weight.grad.shape}")
            # print("Parameter: classify.mlp_1.weight:", f"Gradient: {model.classify.mlp_1.weight.grad}")
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         # print(f"Parameter: {name}, Gradient: {param.grad}")
            #         print(f"Parameter: {name}")

            optimizer.step() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            scheduler.step() 
            model.zero_grad()
            
            avg_train_loss = loss.item() / args.train_batch_size
            global_step += 1
            print("Epoch: {}  global_step: {} Average training loss: {:.7f}".format(epoch, global_step, avg_train_loss))
            writer.add_scalar('avg_train_loss', avg_train_loss, global_step=global_step)
            
            if avg_train_loss < min_train_loss:
                min_train_loss = avg_train_loss
                best_step = global_step
        if neg_counts > 1:
            neg_counts -= 1
        # avg_train_loss = total_train_loss / len(train_dataloader)  
        training_time = format_time(time.time() - t0)
        # time.sleep(0.003)
        # print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("Training epcoh took: {:}".format(training_time))
        print('Saveing Model...')
        torch.save(model, args.checkpoint_dir)
    return model, best_step, min_train_loss, t_total

def evaluate(dataset, model, args, device):
    t0 = time.time()
    model.eval()
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.evaluate_batch_size)

    with torch.no_grad():
        predict_labels_sim = []
        predict_labels_classify = []
        true_labels = []
        epoch_iterator = tqdm(dataloader, desc="Iteration")
        des_features = dataset.get_evaluate_des_features().to(device)
        model.gen_des_vectors(des_features)

        for step, batch in enumerate(epoch_iterator):
            inputs = {'sen_input_ids':      batch["input_ids"].to(device),
                    'sen_att_masks':   batch["attention_mask"].to(device),
                    'marked_e1':        batch["marked_e1"].to(device),
                    'marked_e2':        batch["marked_e2"].to(device),
                    'mark_head':        batch["mark_head"].to(device),
                    'mark_tail':        batch["mark_tail"].to(device)
                    }
            # print('inputs',inputs)
            # logger.info("batch: {}".format(batch))
            outputs = model(**inputs)
            _, max_sim_idx, max_classify_idx = outputs
            predict_labels_sim.extend(max_sim_idx.tolist())
            predict_labels_classify.extend(max_classify_idx.tolist())

            rid_nums = ["P" + str(int(t[0])) for t in batch["rid"]]
            true_label = [dataset.convert_rid_to_label(r) for r in rid_nums] 
            true_labels.extend(true_label)
            # print(predict_labels_classify)
            # print(true_labels)

           
        p_macro_sim, r_macro_sim, f_macro_sim = compute_macro_PRF(np.array(predict_labels_sim), np.array(true_labels))
        p_macro_classify, r_macro_classify, f_macro_classify = compute_macro_PRF(np.array(predict_labels_classify), np.array(true_labels))
        return p_macro_sim, r_macro_sim, f_macro_sim, p_macro_classify, r_macro_classify, f_macro_classify
    
def evaluate_test(dataset, model, args, device):
    t0 = time.time()
    model.eval()
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.evaluate_batch_size)
    tokenizer = AutoTokenizer.from_pretrained("../../bert-base-uncased")
    with torch.no_grad():
        sen_input = []
        predict_sim = []
        predict_labels_sim = []
        predict_labels_classify = []
        true_labels = []
        epoch_iterator = tqdm(dataloader, desc="Iteration")
        des_features = dataset.get_evaluate_des_features().to(device)
        model.gen_des_vectors(des_features)

        for step, batch in enumerate(epoch_iterator):
            inputs = {'sen_input_ids':      batch["input_ids"].to(device),
                    'sen_att_masks':   batch["attention_mask"].to(device),
                    'marked_e1':        batch["marked_e1"].to(device),
                    'marked_e2':        batch["marked_e2"].to(device),
                    'mark_head':        batch["mark_head"].to(device),
                    'mark_tail':        batch["mark_tail"].to(device)
                    }
            # print('inputs',inputs)
            # logger.info("batch: {}".format(batch))

            for single_input_ids in inputs['sen_input_ids']:
                decoded_sentence = tokenizer.decode(single_input_ids, skip_special_tokens=True)
                sen_input.append(decoded_sentence)
                # print(f"sen_input: {decoded_sentence}")

            outputs = model(**inputs)
            max_sim_idx, max_classify_idx = outputs
            predict_labels_sim.extend(max_sim_idx.tolist())
            predict_labels_classify.extend(max_classify_idx.tolist())

            rid_nums = ["P" + str(int(t[0])) for t in batch["rid"]]
            true_label = [dataset.convert_rid_to_label(r) for r in rid_nums] 
            true_labels.extend(true_label)
            # print(predict_labels_classify)
            # print(true_labels)

        # # 将列表转换为张量
        # true_labels = torch.tensor(true_labels)
        # predictions = torch.tensor(predict_labels_sim)

        # # 获取所有类别
        # classes = torch.unique(true_labels).tolist()

        # # 初始化字典来存储每个类别的正确预测数和总数
        # correct_counts = defaultdict(int)
        # total_counts = defaultdict(int)

        # # 计算每个类别的正确预测数和总数
        # for true, pred in zip(true_labels, predictions):
        #     total_counts[true.item()] += 1
        #     if true.item() == pred.item():
        #         correct_counts[true.item()] += 1

        # # 计算每个类别的准确性
        # accuracies = []
        # for cls in classes:
        #     if total_counts[cls] > 0:
        #         accuracy = correct_counts[cls] / total_counts[cls]
        #     else:
        #         accuracy = 0.0
        #     accuracies.append(accuracy)

        # # 打印每个类别的准确性（可选）
        # for cls, acc in zip(classes, accuracies):
        #     print(f"Class {cls}: Accuracy = {acc:.2f}")

        # # 绘制散点图
        # plt.figure(figsize=(10, 6))
        # plt.scatter(classes, accuracies, color='blue', edgecolors='k')

        # # 添加标题和标签
        # plt.title('Model Prediction Accuracy per Class')
        # plt.xlabel('Class')
        # plt.ylabel('Accuracy')

        # # 设置y轴范围为0到1
        # plt.ylim(0, 1)

        # # 添加网格线（可选）
        # plt.grid(True, linestyle='--', alpha=0.5)

        # # 如果类标签是类别名称，可以替换x轴刻度
        # # class_names = ['Class A', 'Class B', 'Class C', 'Class D', 'Class E']
        # # plt.xticks(classes, class_names)

        # # 显示每个点的准确性值
        # for cls, acc in zip(classes, accuracies):
        #     plt.text(cls, acc, f"{acc:.2f}", fontsize=9, ha='right', va='bottom')



        # # 初始化每个标签的预测 distribution
        # pre_distributions = {i: [] for i in range(15)}
        # distributions = {i: [] for i in range(15)}

        # # 统计每种 true_label 值对应的 prediction 分布
        # for i in range(len(true_labels)):
        #     label = true_labels[i]
        #     pred = predict_labels_sim[i]
        #     # sen = sen_input[i]
        #     # if label == 7 and pred !=7:
        #     #     pre_distributions[pred].append([sen])
        #     distributions[label].append(pred)

      
        # # 为散点图准备数据
        # x = []
        # y = []
        # colors = []
        # for label in pre_distributions:
        #     for pred_value in pre_distributions[label]:
        #         x.append(label)  # true_label 值用作 x 轴
        #         y.append(pred_value)  # 对应的 prediction 值用作 y 轴
        #         colors.append(label)  # 颜色根据 true_label 进行区分

        # # 转换为 numpy 数组
        # x = np.array(x)
        # y = np.array(y)
        # colors = np.array(colors)

        # # 绘制散点图
        # plt.figure(figsize=(10, 8))
        # scatter = plt.scatter(x, y, c=colors, cmap='viridis', alpha=0.6)
        # plt.xlabel('True Labels')
        # plt.ylabel('Predictions')
        # plt.title('Scatter Plot of Predictions by True Labels')

        # # 设置图例
        # unique_labels = np.unique(true_label)
        # legend_labels = [f'Label {i}' for i in unique_labels]
        # handles = [plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[i], 
        #                     markerfacecolor=scatter.cmap(scatter.norm(unique_labels[i]))) for i in range(len(unique_labels))]
        # plt.legend(handles=handles)

        # plt.xlim(-1, 15)
        # plt.ylim(-1, 15)
        # plt.grid()
        plt.savefig('sim_scatter_plot.png', format='png', dpi=300)
        plt.show()            
        p_macro_sim, r_macro_sim, f_macro_sim = compute_macro_PRF(np.array(predict_labels_sim), np.array(true_labels))
        p_macro_classify, r_macro_classify, f_macro_classify = compute_macro_PRF(np.array(predict_labels_classify), np.array(true_labels))
        return p_macro_sim, r_macro_sim, f_macro_sim, p_macro_classify, r_macro_classify, f_macro_classify

if __name__=='__main__':
    parser = ArgumentParser()

    # hyperparameters
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--gamma", type=float, default=0.06, help="Loss function: margin factor gamma")
    # parser.add_argument("--alpha", type=float, default=0.33,
                        # help="Similarity: balance entity and context weights in single sample")
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--evaluate_batch_size", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=5, help='training epochs')
    parser.add_argument("--max_seq_len", type=int, default=128, help='max sequence length')
    parser.add_argument("--lr", type=float, default=2e-6, help='learning rate')
    parser.add_argument("--k", type=int, default=3, help='Number of classification')

    parser.add_argument("--warm_up", type=float, default=100, help='warm_up steps')
    parser.add_argument("--unseen", type=int, default=15, help='Number of unseen class')
    parser.add_argument("--expand_data", action='store_true', help='expand the input data')
    # parser.add_argument("--entity_way", type=str, choices=['tmp', 'keyword'], default='tmp',
    #                     help='Representation of the described entity')
    
    # file_path
    parser.add_argument("--dataset_path", type=str, default='data', help='where data stored')
    parser.add_argument("--dataset", type=str, default='fewrel', choices=['fewrel', 'wikizsl'],
                        help='original dataset')
    parser.add_argument("--relation_description_processed", type=str,
                        default='relation_description_processed.json',
                        help='relation descriptions marked entity')

    # model and cuda config
    parser.add_argument("--gpu_available", type=str, default='0', help='the device on which this model will run')
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='bert-base-uncased', help='huggingface pretrained model')
    parser.add_argument("--add_auto_match", type=str, default='True', help='')
    args = parser.parse_args()

    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    args.checkpoint_dir = f'Experiment/ir_ot3_4_{args.dataset}_split_{args.seed}_unseen_{str(args.unseen)}.pth'

    args.data_file = os.path.join(args.dataset_path, args.dataset, f'{args.dataset}_dataset.json')
    args.relation_description_file = os.path.join(args.dataset_path, args.dataset, 'relation_description',
                                                  f'{args.dataset}_relation_description.json')
    args.relation_description_file_processed = os.path.join(args.dataset_path, args.dataset, 'relation_description',
                                                args.relation_description_processed)
    
    add_auto_match = True if args.add_auto_match == 'True' else False

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_time = time.time()
    # if not os.path.exists(args.checkpoint_dir):
    #     os.makedirs(args.checkpoint_dir)
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning(f'device: {args.gpu_available}, epochs: {args.epochs}, lr: {args.lr}, seed: {args.seed}, batch size: {args.train_batch_size}')

    # set seed
    set_seed(args.seed)
    
    args.device = torch.device("cuda:" + args.gpu_available if torch.cuda.is_available() else "cpu")
    
    model = EMMA(args.pretrained_model_name_or_path, add_auto_match, args.max_seq_len, args.k)
    model.to(args.device)

    # train
    train_dataset = Dataset("train", args.data_file, args.relation_description_file, 
                            args.relation_description_file_processed, args.unseen,
                            args.pretrained_model_name_or_path, args.max_seq_len, model, args, expand_or_not=args.expand_data)
    # model, best_step, min_train_loss, total_steps= train(train_dataset, model, args, args.device)
    
    # model = torch.load("checkpoints/ir_ot3_2_fewrel_split_7_unseen_15.pth")
    model = torch.load("Experiment/model_fewrel_split_7_unseen_15.pth")
    
    # dev
    train_dataset.mode = "dev"
    dev_dataset = train_dataset
    # p_macro_sim, r_macro_sim, f_macro_sim, p_macro_classify, r_macro_classify, f_macro_classify = evaluate(dev_dataset, model, args, args.device)
    # dev_info_sim = f'[dev][sim] (macro) final precision: {p_macro_sim:.4f}, recall: {r_macro_sim:.4f}, f1 score: {f_macro_sim:.4f}'
    # print(dev_info_sim)
    # dev_info_classify = f'[dev][classify] (macro) final precision: {p_macro_classify:.4f}, recall: {r_macro_classify:.4f}, f1 score: {f_macro_classify:.4f}'
    # print(dev_info_classify)

    # test
    dev_dataset.mode = "test"
    test_dataset = dev_dataset
    p_macro_sim, r_macro_sim, f_macro_sim, p_macro_classify, r_macro_classify, f_macro_classify = evaluate_test(test_dataset, model, args, args.device)
    test_info_sim = f'[test][sim] (macro) final precision: {p_macro_sim:.4f}, recall: {r_macro_sim:.4f}, f1 score: {f_macro_sim:.4f}'
    print(test_info_sim)
    test_info_classify = f'[test][classify] (macro) final precision: {p_macro_classify:.4f}, recall: {r_macro_classify:.4f}, f1 score: {f_macro_classify:.4f}'
    print(test_info_classify)
    # running time
    end_time = time.time()
    run_time = end_time - start_time
    with open("result_model_recall_ir_ot3_2_1.6_fewrel.txt", "a") as file:
    # with open("result_model_recall_ir_ot3_2_12.19_new.txt", "a") as file:
        # file.write("w/0: " + "sen_between_vec" + "\n")
        # file.write("Datetime: " + current_datetime + "\n")
        # file.write("Run time: {:.2f} seconds\n".format(run_time))
        file.write(f"Total steps: {total_steps}\n")
        file.write(f"Best step: {best_step}\n")
        file.write(f"Min train loss: {min_train_loss}\n")
        file.write("Parameters info:\n")
        for arg in vars(args):
            file.write(f"\t {arg}: {getattr(args, arg)}\n")
        file.write("\n")
        file.write("Evaluation results:\n")
        file.write(dev_info_sim + "\n")
        file.write(dev_info_classify + "\n")
        file.write(test_info_sim + "\n")
        file.write(test_info_classify + "\n")
        file.write("\n")
        