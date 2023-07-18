import torch
import numpy as np
from torchmetrics.classification import MulticlassAccuracy
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import pandas as pd
from termcolor import colored
import seaborn as sns
import matplotlib.pyplot as plt


def one_of_k_encoding(x, allowable_set):
  if x not in allowable_set:
    raise Exception("input {0} not in allowable set{1}:".format(
        x, allowable_set))
  return list(map(lambda s: x == s, allowable_set))
 
def one_of_k_encoding_unk(x, allowable_set):
  """Maps inputs not in the allowable set to the last element."""
  if x not in allowable_set:
    x = allowable_set[-1]
  return list(map(lambda s: x == s, allowable_set))




def accuracy(out, labels):
    # outputs = np.argmax(out, axis=1)
    # print(f"shapes = {out.shape}, {labels.shape}")
    if len(out.shape) > 1:
        out = out.reshape(-1)
    if len(labels.shape) > 1:
        labels = labels.reshape(-1)
    if labels.shape[0] == 0:
        return 0
    return torch.sum(out == labels) / labels.shape[0]

def classwise_accuracy(out, 
                       labels,
                       num_classes):
    if len(out.shape) > 1:
        out = out.reshape(-1)
    if len(labels.shape) > 1:
        labels = labels.reshape(-1)
    for i in range(num_classes):
        class_mask = labels == i
        class_acc = accuracy(out[class_mask], labels[class_mask])
        print(f"Class {i} accuracy: {class_acc}")

def predicted_start_and_end(out, labels):
    assert len(out.shape) == 2
    assert len(labels.shape) == 2
    # finding the begin label: 0, in labels
    # and the end label: 3, in labels

    begin_gts = torch.zeros(0, dtype=torch.long).to(labels.device)
    end_gts = torch.zeros(0, dtype=torch.long).to(labels.device)

    for i in range(labels.shape[0]):
        begin_gts = torch.cat((begin_gts, 
                               torch.where(labels[i] == 0)[0][0].reshape(-1)))
        end_gts = torch.cat((end_gts, 
                             torch.where(labels[i] == 3)[0][-1].reshape(-1)))

def accuracy_of_masked_part(out, 
                            labels,
                            mask_start_idx,
                            mask_end_idx):
    assert len(out.shape) == 2
    assert len(labels.shape) == 2
    # print(f"mask_start_idx: {mask_start_idx.shape}")
    # print(f"mask_end_idx: {mask_end_idx.shape}")
    if type(mask_start_idx) != torch.LongTensor:
        mask_start_idx = mask_start_idx.to(torch.long).to(labels.device).reshape(-1)
    if type(mask_end_idx) != torch.LongTensor:
        mask_end_idx = mask_end_idx.to(torch.long).to(labels.device).reshape(-1)
        # convert to int
    # rows = torch.arange(labels.shape[0]).to(labels.device).reshape(-1, 1)
    total_correct = 0
    total_count = 0

    for row in range(labels.shape[0]):
        req_label = labels[row, mask_start_idx[row]:mask_end_idx[row]+1]
        pred_label = out[row, mask_start_idx[row]:mask_end_idx[row]+1]
        req_label = req_label.reshape(-1)
        pred_label = pred_label.reshape(-1)
        total_correct += torch.sum(req_label == pred_label)
        total_count += req_label.shape[0]
    
    accuracy = total_correct / total_count

    return accuracy

def get_start_and_end_idx(antibody_masked_token,
                          antibody_full_token,
                          model_output,
                          config,
                          token_ids):
    # model_output: (batch_size, num_classes, seq_len)
    mask_token_id = token_ids[config['mask_token']]
    total_masked_out = torch.zeros((0), dtype=torch.long).to(model_output.device)
    total_masked_gts = torch.zeros((0), dtype=torch.long).to(model_output.device)

    for j in range(antibody_masked_token.shape[0]):
        start_idx = -1
        end_idx = -1

        for i in range(antibody_masked_token[j].shape[0]):
            if antibody_masked_token[j, i] == mask_token_id:
                if start_idx == -1:
                    start_idx = i
                else:
                    end_idx = i
            elif antibody_masked_token[j, i] != mask_token_id and end_idx != -1:
                break
        
        req_full_token = antibody_full_token[j, start_idx:end_idx+1]
        req_model_output = model_output[j, :, start_idx:end_idx+1]
        req_model_output = torch.permute(req_model_output, (1, 0))
        total_masked_out = torch.cat((total_masked_out, torch.argmax(req_model_output, 1).reshape(-1)), dim=0)
        total_masked_gts = torch.cat((total_masked_gts, req_full_token.reshape(-1)), dim=0)
    # assert total_masked_gts.shape == total_masked_gts.shape
    return total_masked_out, total_masked_gts

def torch_metrics_accuracy(preds,
                           targets,
                           ignore_idx,
                           num_classes):
    preds = preds.cpu()
    targets = targets.cpu()
    accuracy_obj = MulticlassAccuracy(num_classes, 
                                      average='micro', 
                                      ignore_index=ignore_idx)
    accuracy = accuracy_obj(preds, targets)
    return accuracy

def torch_metric_accuracy_topk(preds,
                               targets,
                               topk,
                               ignore_idx,
                               num_classes):
    preds = preds.cpu()
    targets = targets.cpu()
    accuracy_obj = MulticlassAccuracy(num_classes, 
                                      topk=topk, 
                                      average='micro', 
                                      ignore_index=ignore_idx,
                                      validate_args=True)
    accuracy = accuracy_obj(preds, targets)
    return accuracy

def my_topk_accuracy(preds,
                     targets,
                     topk,
                     ignore_idx,
                     num_classes):
    
    topk_indices = torch.topk(preds, topk, dim=1)[1]
    topk_indices = topk_indices.cpu()
    targets = targets.cpu()
    total_correct = 0
    total_count = 0
    for i in range(targets.shape[0]):
        if targets[i] == ignore_idx:
            continue
        if targets[i] in topk_indices[i]:
            total_correct += 1
        total_count += 1
    if total_count == 0:
        print("debug target")
        print(targets)
    accuracy = total_correct / total_count
    return accuracy

# for i in range(len(antibody_masked_token)):
#             if antibody_masked_token[i] == mask_token_id:
#                 if start_idx == -1:
#                     start_idx = i
#                 else:
#                     end_idx = i
#             elif antibody_masked_token[i] != mask_token_id and end_idx != -1:
#                 break
        
#         req_full_token = antibody_full_token[start_idx:end_idx+1]
#         req_model_output = model_output[:,  :, start_idx:end_idx+1]


def multilabel_accuracy(preds, 
                       targets):
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()

    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    targets_ones = np.where(targets == 1)[0]
    correct_num = (preds[targets_ones] == targets[targets_ones]).sum()
    total_num = targets_ones.shape[0]
    accuracy = correct_num / total_num
    return accuracy

def confusion_matrix(preds, 
                     targets,
                     num_classes):
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    matrix = multilabel_confusion_matrix(targets, preds)
    print(matrix)
    # correct_num = (preds == targets).sum()
    # total_num = targets.shape[0]
    # accuracy = correct_num / total_num
    # return accuracy
    
def calculate_num_data_per_class(np_data,
                                 class_names,
                                 class_num=11):
    print(f"Distribution of data per class")
    class_num_dict = {}
    for i in range(class_num):
        class_num_dict[i] = 0

    for i in range(np_data.shape[0]):
        labels = np_data[i]['label']
        if type(labels) != list:
            labels = [labels]
        for label in labels:
            class_num_dict[int(label)] += 1

    for class_name, count in zip(class_names.values(), 
                                 class_num_dict.values()):
        print(f"{class_name}: {count}")

    class_names_count = {}
    for class_name, count in zip(class_names.values(),
                                 class_num_dict.values()):
        class_names_count[class_name] = count

    return class_names_count

def biased_accuracy(preds,
                    targets,
                    threshold=0.8):
    preds = preds.detach().cpu().numpy()
    targets = targets.cpu().numpy() #N*1

    # preds is 1 if the probability of positive is greater than threshold
    preds = (preds[:, 1] > threshold).astype(int)
    targets = targets.reshape(-1)
    preds = preds.reshape(-1)
    confusion_mat = sk_confusion_matrix(targets, preds)
    correct_num = (preds == targets).sum()
    total_num = targets.shape[0]
    accuracy = correct_num / total_num
    return accuracy, confusion_mat


def make_colored_string(input_string,
                        index_to_color):
    colored_indices = {index: "red" for index in index_to_color}
    colored_string = [colored(input_string[i], colored_indices[i]) if i in colored_indices else input_string[i] for i in range(len(input_string))]
    colored_string = "".join(colored_string)
    return colored_string



