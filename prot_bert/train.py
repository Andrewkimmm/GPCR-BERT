import torch
import torch.nn as nn
import numpy as np
from utils import accuracy, \
                  classwise_accuracy, \
                  get_start_and_end_idx, \
                  torch_metrics_accuracy, \
                  torch_metric_accuracy_topk, \
                  my_topk_accuracy, \
                  multilabel_accuracy, \
                  confusion_matrix, \
                  biased_accuracy
from tqdm import tqdm
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


def train_forward_two_position_prediction(model,
                                          data_loader,
                                          optimizer,
                                          loss_fn,
                                          device,
                                          config):
    '''
    Train function for forward to predict two positions
    Uses ClassificationTwoPosition model from model.py
    '''
    model = model.train()
    total_loss = 0
    total_count = 0
    total_gts = torch.zeros((0), dtype=torch.long).to(device)
    # total_preds = torch.zeros((0), dtype=torch.long).to(device)
    total_logits = torch.zeros((0, config['ClassificationTwoPosition']['FeedForward']['num_classes']), 
                               dtype=torch.float).to(device)
    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='green')
    
    for idx, batch in enumerate(loop):
        input_tokens, \
        label_tokens, \
        start_idx, \
        end_idx = batch


        print("idx", idx)
        input_tokens['input_ids'] = input_tokens['input_ids'].to(device).squeeze(1)
        input_tokens['token_type_ids'] = input_tokens['token_type_ids'].to(device).squeeze(1)
        input_tokens['attention_mask'] = input_tokens['attention_mask'].to(device).squeeze(1)

        label_tokens['input_ids'] = label_tokens['input_ids'].to(device).squeeze(1)
        label_tokens['token_type_ids'] = label_tokens['token_type_ids'].to(device).squeeze(1)
        label_tokens['attention_mask'] = label_tokens['attention_mask'].to(device).squeeze(1)

        if config['ClassificationTwoPosition']['use_bert']:
            out = model(input_tokens)
        else:
            out = model(input_tokens['input_ids'],
                        input_tokens['attention_mask']) # (batch_size, class_num, seq_len)
        iteration_loss = loss_fn(out, label_tokens['input_ids'])

        optimizer.zero_grad()
        iteration_loss.backward()
        optimizer.step()

        total_loss += iteration_loss.item() * input_tokens['input_ids'].shape[0]
        total_count += input_tokens['input_ids'].shape[0]
        out = out.permute(0, 2, 1)
       
        for b_idx in range(out.shape[0]):
            # curr_start_idx = start_idx[b_idx]
            # total_logits = torch.cat((total_logits,
            #                           out[b_idx, curr_start_idx, :].reshape(1, -1),
            #                           out[b_idx, curr_start_idx+1, :].reshape(1, -1)),
            #                          dim=0) # (batch_size*2, class_num)
            # total_gts = torch.cat((total_gts,
            #                        label_tokens['input_ids'][b_idx, curr_start_idx].reshape(1),
            #                        label_tokens['input_ids'][b_idx, curr_start_idx+1].reshape(1)),
            #                        dim=0) # (batch_size*2)
            
            for seq_idx in range(start_idx[b_idx], 
                                 end_idx[b_idx]):
                # print(seq_idx)
                total_logits = torch.cat((total_logits,
                                          out[b_idx, seq_idx, :].reshape(1, -1)),
                                         dim=0)
                total_gts = torch.cat((total_gts,
                                       label_tokens['input_ids'][b_idx, seq_idx].reshape(1)),
                                      dim=0)
                # print(label_tokens['input_ids'][b_idx, seq_idx])

        loop.set_description(f"Loss: {iteration_loss.item()}")

    # print(total_gts.shape)
    # print(total_logits.shape)
    total_preds = torch.argmax(total_logits, dim=1)
    train_accuracy = torch_metrics_accuracy(total_preds.reshape(-1),
                                            total_gts.reshape(-1),
                                            num_classes=config['ClassificationTwoPosition']['FeedForward']['num_classes'],
                                            ignore_idx=0)

    for k in range(1, config['topk']):
        train_topk_accuracy = my_topk_accuracy(total_logits,
                                               total_gts.reshape(-1),
                                               topk=k,
                                               ignore_idx=0,
                                               num_classes=config['ClassificationTwoPosition']['FeedForward']['num_classes'])
        print(f"Train Top-{k} Accuracy: {train_topk_accuracy}")

    return train_accuracy, total_loss / total_count, 0.0

def test_forward_two_position_prediction(model,
                                         data_loader,
                                         loss_fn,
                                         device,
                                         config):
    '''
    Corresponding test function for forward to predict two positions
    '''
    model = model.eval()
    total_loss = 0
    total_count = 0
    total_gts = torch.zeros((0), dtype=torch.long).to(device)
    total_preds = torch.zeros((0), dtype=torch.long).to(device)
    total_logits = torch.zeros((0, config['ClassificationTwoPosition']['FeedForward']['num_classes']),
                                dtype=torch.float).to(device)
    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='green')

    for idx, batch in enumerate(loop):
        input_tokens, \
        label_tokens, \
        start_idx, \
        end_idx = batch

        input_tokens['input_ids'] = input_tokens['input_ids'].to(device).squeeze(1)
        input_tokens['token_type_ids'] = input_tokens['token_type_ids'].to(device).squeeze(1)
        input_tokens['attention_mask'] = input_tokens['attention_mask'].to(device).squeeze(1)

        label_tokens['input_ids'] = label_tokens['input_ids'].to(device).squeeze(1)
        label_tokens['token_type_ids'] = label_tokens['token_type_ids'].to(device).squeeze(1)
        label_tokens['attention_mask'] = label_tokens['attention_mask'].to(device).squeeze(1)

        with torch.no_grad():
            if config['ClassificationTwoPosition']['use_bert']:
                out = model(input_tokens)
            else:
                out = model(input_tokens['input_ids'],
                            input_tokens['attention_mask'])
            iteration_loss = loss_fn(out, label_tokens['input_ids'])

        total_loss += iteration_loss.item() * input_tokens['input_ids'].shape[0]
        total_count += input_tokens['input_ids'].shape[0]
        out = out.permute(0, 2, 1)

        for b_idx in range(out.shape[0]):
            # curr_start_idx = start_idx[b_idx]
            # total_logits = torch.cat((total_logits,
            #                           out[b_idx, curr_start_idx, :].reshape(1, -1),
            #                           out[b_idx, curr_start_idx+1, :].reshape(1, -1)),
            #                          dim=0)
            # total_gts = torch.cat((total_gts,
            #                        label_tokens['input_ids'][b_idx, curr_start_idx].reshape(1),
            #                        label_tokens['input_ids'][b_idx, curr_start_idx+1].reshape(1)),
            #                       dim=0)

            for seq_idx in range(start_idx[b_idx], 
                                 end_idx[b_idx]):
                total_logits = torch.cat((total_logits,
                                          out[b_idx, seq_idx, :].reshape(1, -1)),
                                         dim=0)
                total_gts = torch.cat((total_gts,
                                       label_tokens['input_ids'][b_idx, seq_idx].reshape(1)),
                                      dim=0)
    
       
        loop.set_description(f"Loss: {iteration_loss.item()}")
    total_preds = torch.argmax(total_logits, dim=1)
    test_accuracy = torch_metrics_accuracy(total_preds.reshape(-1),
                                           total_gts.reshape(-1),
                                           num_classes=config['ClassificationTwoPosition']['FeedForward']['num_classes'],
                                           ignore_idx=0)

    for k in range(1, config['topk']):
        test_topk_accuracy = my_topk_accuracy(total_logits,
                                              total_gts.reshape(-1),
                                              topk=k,
                                              ignore_idx=0,
                                              num_classes=config['ClassificationTwoPosition']['FeedForward']['num_classes'])
        print(f"Test Top-{k} Accuracy: {test_topk_accuracy}")
    
    final_pred = total_preds.reshape(-1)
    final_gts = total_gts.reshape(-1)

    return test_accuracy, total_loss / total_count, 0.0, final_pred, final_gts
        


if __name__ == "__main__":
    pass
