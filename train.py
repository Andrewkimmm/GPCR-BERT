import torch
import torch.nn as nn
import numpy as np
import wandb
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



def train_forward(model,
                  data_loader,
                  optimizer,
                  loss_fn,
                  device,
                  scaler=None,
                  config=None):
    model.train()
    total_loss = 0
    total_count = 0
    total_gts = torch.zeros(0, dtype=torch.long).to(device)
    total_preds = torch.zeros(0, dtype=torch.long).to(device)
    loop = tqdm(data_loader, 
                leave=True, 
                total=len(data_loader),
                colour='green')

    for idx, batch in enumerate(loop):
        virus_token_ids, virus_pos_ids, virus_attention_mask, \
        antibody_token_ids, antibody_pos_ids, antibody_attention_mask, \
        cross_attention_mask, classes, \
        mask_start_idx, mask_end_idx = batch

        antibody_token_ids = antibody_token_ids.to(device)
        antibody_pos_ids = antibody_pos_ids.to(device)
        antibody_attention_mask = antibody_attention_mask.to(device)
        cross_attention_mask = cross_attention_mask.to(device)
        classes = classes.to(device)
        mask_start_idx = mask_start_idx.to(device)
        mask_end_idx = mask_end_idx.to(device)
        virus_token_ids = virus_token_ids.to(device)
        virus_pos_ids = virus_pos_ids.to(device)
        virus_attention_mask = virus_attention_mask.to(device)
    
        out = model(virus_token_ids,
                    virus_pos_ids,
                    virus_attention_mask,
                    antibody_token_ids,
                    antibody_pos_ids,
                    antibody_attention_mask,
                    cross_attention_mask = cross_attention_mask if config['use_cross_attention_mask'] else None) # Batch * C * seq_len

        classes_req = torch.zeros((0), dtype=torch.long).to(device)
        out_req = torch.zeros((0, out.shape[1])).to(device)
        for row in range(classes.shape[0]):
            classes_req = torch.cat((classes_req, 
                                     classes[row, mask_start_idx[row]: mask_end_idx[row]+1].reshape(-1)),
                                     dim=0)
            out_slice = out[row, :, mask_start_idx[row]: mask_end_idx[row]+1]
            out_slice = torch.permute(out_slice, (1, 0))
            out_req = torch.cat((out_req, out_slice), dim=0)
        out = out_req
        classes = classes_req

        iteration_loss = loss_fn(out, classes)
        optimizer.zero_grad()
        iteration_loss.backward()
        optimizer.step()
            
        total_loss += iteration_loss.item() * virus_token_ids.shape[0]
        total_count += virus_token_ids.shape[0]
        total_gts = torch.cat((total_gts, 
                               classes), axis=0)
        total_preds = torch.cat((total_preds, 
                                 torch.argmax(out, dim=1, keepdim=False)))
       
        loop.set_description(f"Loss: {iteration_loss.item()}")
        # break
    train_accuracy = accuracy(total_preds, total_gts)
    train_loss = total_loss / total_count
    masked_train_accuracy = train_accuracy
    wandb.log({"train_masked_accuracy": masked_train_accuracy,
                "train_accuracy": train_accuracy,
                "train_loss": train_loss})
    return train_accuracy, train_loss, masked_train_accuracy
    return train_accuracy, train_loss

def test_forward(model,
                 data_loader,
                 loss_fn,
                 device,
                 scaler=None,
                 config=None):
    model.eval()
    total_loss = 0
    total_count = 0
    total_gts = torch.zeros(0, dtype=torch.long).to(device)
    total_preds = torch.zeros(0, dtype=torch.long).to(device)

    loop = tqdm(data_loader, 
                leave=True, 
                total=len(data_loader),
                colour='blue')

    with torch.no_grad():
        for idx, batch in enumerate(loop):
            virus_token_ids, \
            virus_pos_ids, \
            virus_attention_mask, \
            antibody_token_ids, \
            antibody_pos_ids, \
            antibody_attention_mask, \
            cross_attention_mask, \
            classes, \
            mask_start_idx, \
            mask_end_idx = batch

            virus_token_ids = virus_token_ids.to(device)
            virus_pos_ids = virus_pos_ids.to(device)
            virus_attention_mask = virus_attention_mask.to(device)
            antibody_token_ids = antibody_token_ids.to(device)
            antibody_pos_ids = antibody_pos_ids.to(device)
            antibody_attention_mask = antibody_attention_mask.to(device)
            cross_attention_mask = cross_attention_mask.to(device)
            classes = classes.to(device)
            mask_start_idx = mask_start_idx.to(device)
            mask_end_idx = mask_end_idx.to(device)

            out = model(virus_token_ids,
                        virus_pos_ids,
                        virus_attention_mask,
                        antibody_token_ids,
                        antibody_pos_ids,
                        antibody_attention_mask,
                        cross_attention_mask = cross_attention_mask if config['use_cross_attention_mask'] else None)
            classes_req = torch.zeros((0), dtype=torch.long).to(device)
            out_req = torch.zeros((0, out.shape[1])).to(device)
            for row in range(classes.shape[0]):
                classes_req = torch.cat((classes_req, 
                                            classes[row, mask_start_idx[row]: mask_end_idx[row]+1].reshape(-1)),
                                            dim=0)
                out_slice = out[row, :, mask_start_idx[row]: mask_end_idx[row]+1]
                out_slice = torch.permute(out_slice, (1, 0))
                out_req = torch.cat((out_req, out_slice), dim=0)
            out = out_req
            classes = classes_req
            iteration_loss = loss_fn(out, classes)

            total_loss += iteration_loss.item() * virus_token_ids.shape[0]
            total_count += virus_token_ids.shape[0]
            total_gts = torch.cat((total_gts, 
                                   classes), dim=0)
            total_preds = torch.cat((total_preds, 
                                     torch.argmax(out, dim=1, keepdim=False)))
    
            loop.set_description(f"Loss: {iteration_loss.item()}")
    
    test_accuracy = accuracy(total_preds, total_gts)
    classwise_accuracy(total_preds, total_gts, config['vocabulary_size'])
    test_loss = total_loss / total_count
    masked_test_accuracy = test_accuracy
    wandb.log({'test_masked_accuracy': masked_test_accuracy,
                'test_accuracy': test_accuracy,
                'test_loss': test_loss})
    return test_accuracy, test_loss, masked_test_accuracy
    return test_accuracy, test_loss

def train_forward_masked(model,
                         data_loader,
                         optimizer,
                         loss_fn,
                         device,
                         config=None):
    '''
    Train forward pass for masked language model, which use
    torch.nn.TransformerEncoder as the encoder. and predicts
    the masked cdr region
    Uses Model class from model.py
    '''
    model = model.train()
    total_loss = 0
    total_count = 0
    num_classes = config['FeedForward']['num_classes']
    total_gts = torch.zeros((0), dtype=torch.long).to(device)
    total_preds = torch.zeros((0), dtype=torch.long).to(device)
    total_mask_preds = torch.zeros((0), dtype=torch.float).to(device)
    total_mask_gts = torch.zeros((0), dtype=torch.float).to(device)

    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='green')
    
    for idx, batch in enumerate(loop):
        virus_token_ids, \
        virus_pos_ids, \
        virus_attention_mask, \
        antibody_token_ids, \
        antibody_pos_ids, \
        antibody_attention_mask, \
        cross_attention_mask, \
        antibody_masked_token_ids = batch

        virus_token_ids = virus_token_ids.to(device)
        virus_pos_ids = virus_pos_ids.to(device)
        virus_attention_mask = virus_attention_mask.to(device)
        antibody_token_ids = antibody_token_ids.to(device)
        antibody_pos_ids = antibody_pos_ids.to(device)
        antibody_attention_mask = antibody_attention_mask.to(device)
        cross_attention_mask = cross_attention_mask.to(device)
        antibody_masked_token_ids = antibody_masked_token_ids.to(device)
        # print(f"antibody_masked_token: {antibody_masked_token_ids[0]}")
        # print(f"antibody_token: {antibody_token_ids.shape}")
        # print(f"max and min in gt token ids: {torch.max(antibody_token_ids)}, {torch.min(antibody_token_ids)}")
        # print(f"max and min in masked token ids: {torch.max(antibody_masked_token_ids)}, {torch.min(antibody_masked_token_ids)}")

        out = model(virus_token_ids,
                    virus_pos_ids,
                    virus_attention_mask,
                    antibody_masked_token_ids,
                    antibody_pos_ids,
                    antibody_attention_mask,
                    cross_attention_mask)
        
        if config['measure_masked_accuracy']:
            masked_out, masked_gt = get_start_and_end_idx(antibody_masked_token_ids,
                                                          antibody_token_ids,
                                                          out,
                                                          config,
                                                          config['token_ids'])
            total_mask_gts = torch.cat((total_mask_gts, masked_gt.reshape(-1)), dim=0)
            total_mask_preds = torch.cat((total_mask_preds, 
                                          masked_out.reshape(-1)), dim=0)
        # print(f"out shape: {out}")
        iteration_loss = loss_fn(out, antibody_token_ids)
        
        optimizer.zero_grad()
        iteration_loss.backward()
        optimizer.step()

        total_loss += iteration_loss.item() * virus_token_ids.shape[0]
        total_count += virus_token_ids.shape[0]
        # print(antibody_token_ids.shape)
        total_gts = torch.cat((total_gts,
                               antibody_token_ids.reshape(-1)), dim=0)
        out = torch.permute(out, (0, 2, 1))

        total_preds = torch.cat((total_preds,
                                 torch.argmax(out, dim=2, keepdim=False).reshape(-1 )), dim=0)
        
        loop.set_description(f"Loss: {iteration_loss.item()}")
    
    # train_accuracy = accuracy(total_preds, total_gts)
    train_accuracy = torch_metrics_accuracy(total_preds, total_gts, ignore_idx=0)
    # classwise_accuracy(total_preds, total_gts, config['FeedForward']['num_classes'])
    if total_mask_gts.shape[0] != 0:
        masked_train_accuracy = accuracy(total_mask_preds, total_mask_gts)
        wandb.log({'train_masked_accuracy': masked_train_accuracy})

    train_loss = total_loss / total_count
    wandb.log({'train_accuracy': train_accuracy,
               'train_loss': train_loss})
    return train_accuracy, train_loss, masked_train_accuracy

def test_forward_masked(model,
                        data_loader,
                        loss_fn,
                        device,
                        config=None):
    ''' 
    corresponding test function for train_forward_masked
    '''
    model = model.eval()
    total_loss = 0
    total_count = 0
    num_classes = config['FeedForward']['num_classes']
    total_gts = torch.zeros((0), dtype=torch.long).to(device)
    total_preds = torch.zeros((0), dtype=torch.long).to(device)
    total_mask_preds = torch.zeros((0), dtype=torch.float).to(device)
    total_mask_gts = torch.zeros((0), dtype=torch.float).to(device)
    
    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='green')
    
    for idx, batch in enumerate(loop):
        virus_token_ids, \
        virus_pos_ids, \
        virus_attention_mask, \
        antibody_token_ids, \
        antibody_pos_ids, \
        antibody_attention_mask, \
        cross_attention_mask, \
        antibody_masked_token_ids = batch

        virus_token_ids = virus_token_ids.to(device)
        virus_pos_ids = virus_pos_ids.to(device)
        virus_attention_mask = virus_attention_mask.to(device)
        antibody_token_ids = antibody_token_ids.to(device)
        antibody_pos_ids = antibody_pos_ids.to(device)
        antibody_attention_mask = antibody_attention_mask.to(device)
        cross_attention_mask = cross_attention_mask.to(device)
        antibody_masked_token_ids = antibody_masked_token_ids.to(device)

        with torch.no_grad():
            out = model(virus_token_ids,
                        virus_pos_ids,
                        virus_attention_mask,
                        antibody_masked_token_ids,
                        antibody_pos_ids,
                        antibody_attention_mask,
                        cross_attention_mask)
            iteration_loss = loss_fn(out, antibody_token_ids)
            # out_for_print = torch.permute(out, (0, 2, 1))
            # out_for_print = torch.argmax(out_for_print, 2, keepdims=False)
            # print(f"pred\n: {out_for_print[0]}")
            # print(f"gt\n: {antibody_token_ids[0]}")
            # print(out.shape, antibody_token_ids.shape)

        if config['measure_masked_accuracy']:
            masked_out, masked_gt = get_start_and_end_idx(antibody_masked_token_ids,
                                                            antibody_token_ids,
                                                            out,
                                                            config,
                                                            config['token_ids'])
            total_mask_gts = torch.cat((total_mask_gts, masked_gt.reshape(-1)), dim=0)
            total_mask_preds = torch.cat((total_mask_preds,
                                          masked_out.reshape(-1)), dim=0)
            
        

        total_loss += iteration_loss.item() * virus_token_ids.shape[0]
        total_count += virus_token_ids.shape[0]
        total_gts = torch.cat((total_gts,
                               antibody_token_ids.reshape(-1)), dim=0)
        out = torch.permute(out, (0, 2, 1))
        total_preds = torch.cat((total_preds,
                                 torch.argmax(out, dim=2, keepdim=False).reshape(-1 )), dim=0)
    
        loop.set_description(f"Loss: {iteration_loss.item()}")
    
        test_accuracy = torch_metrics_accuracy(total_preds, total_gts, ignore_idx=0)
        # test_accuracy = accuracy(total_preds, total_gts)

    if total_mask_gts.shape[0] != 0:
        masked_test_accuracy = accuracy(total_mask_preds, total_mask_gts)
        print(f"classwise test mask accuracy")
        classwise_accuracy(total_mask_preds, total_mask_gts, config['FeedForward']['num_classes'])
        wandb.log({'test_masked_accuracy': masked_test_accuracy})

    # print(f"Masked Test Accuracy: {masked_test_accuracy}")
    classwise_accuracy(total_preds, total_gts, config['FeedForward']['num_classes'])
    test_loss = total_loss / total_count
    wandb.log({'test_accuracy': test_accuracy,
                'test_loss': test_loss})
    return test_accuracy, test_loss, masked_test_accuracy

def train_forward_bert(model,
                       data_loader,
                       optimizer,
                       loss_fn,
                       device,
                       config=None):
    '''
    Train forward bert model, where the masked antibody sequence is fed into the model
    and the model predicts masked cdr region.
    Uses BertModel class from model.py
    '''
    model = model.train()
    total_loss = 0
    total_count = 0
    torch_gts = torch.zeros((0), dtype=torch.long).to(device)
    torch_preds = torch.zeros((0), dtype=torch.long).to(device)
    total_preds_logits = torch.zeros((0, config['FeedForward']['num_classes']), 
                                     dtype=torch.float).to(device)

    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='green')
    
    for idx, batch in enumerate(loop):
        bert_antibody_full_tokens, \
        bert_antibody_masked_tokens, \
        bert_virus_full_tokens, \
        cross_attention_mask = batch

        # bert_antibody_full_tokens = bert_antibody_full_tokens.to(device)
        # bert_antibody_masked_tokens = bert_antibody_masked_tokens.to(device)
        # bert_virus_full_tokens = bert_virus_full_tokens.to(device)
        # cross_attention_mask = cross_attention_mask.to(device)

        bert_antibody_full_tokens['input_ids'] = bert_antibody_full_tokens['input_ids'].to(device)
        bert_antibody_full_tokens['token_type_ids'] = bert_antibody_full_tokens['token_type_ids'].to(device)
        bert_antibody_full_tokens['attention_mask'] = bert_antibody_full_tokens['attention_mask'].to(device)
        bert_antibody_masked_tokens['input_ids'] = bert_antibody_masked_tokens['input_ids'].to(device)
        bert_antibody_masked_tokens['token_type_ids'] = bert_antibody_masked_tokens['token_type_ids'].to(device)
        bert_antibody_masked_tokens['attention_mask'] = bert_antibody_masked_tokens['attention_mask'].to(device)
        bert_virus_full_tokens['input_ids'] = bert_virus_full_tokens['input_ids'].to(device)
        bert_virus_full_tokens['token_type_ids'] = bert_virus_full_tokens['token_type_ids'].to(device)
        bert_virus_full_tokens['attention_mask'] = bert_virus_full_tokens['attention_mask'].to(device)
        cross_attention_mask = cross_attention_mask.to(device)

        # squeeze the second dimension
        bert_antibody_full_tokens['input_ids'] = bert_antibody_full_tokens['input_ids'].squeeze(1)
        bert_antibody_full_tokens['token_type_ids'] = bert_antibody_full_tokens['token_type_ids'].squeeze(1)
        bert_antibody_full_tokens['attention_mask'] = bert_antibody_full_tokens['attention_mask'].squeeze(1)
        bert_antibody_masked_tokens['input_ids'] = bert_antibody_masked_tokens['input_ids'].squeeze(1)
        bert_antibody_masked_tokens['token_type_ids'] = bert_antibody_masked_tokens['token_type_ids'].squeeze(1)
        bert_antibody_masked_tokens['attention_mask'] = bert_antibody_masked_tokens['attention_mask'].squeeze(1)
        bert_virus_full_tokens['input_ids'] = bert_virus_full_tokens['input_ids'].squeeze(1)
        bert_virus_full_tokens['token_type_ids'] = bert_virus_full_tokens['token_type_ids'].squeeze(1)
        bert_virus_full_tokens['attention_mask'] = bert_virus_full_tokens['attention_mask'].squeeze(1)

        out = model(bert_antibody_full_tokens,
                    bert_antibody_masked_tokens,
                    bert_virus_full_tokens,
                    cross_attention_mask)
        iteration_loss = loss_fn(out, 
                                 bert_antibody_full_tokens['input_ids'])
        optimizer.zero_grad()
        iteration_loss.backward()
        optimizer.step()

        total_loss += iteration_loss.item() * bert_antibody_full_tokens['input_ids'].shape[0]
        total_count += bert_antibody_full_tokens['input_ids'].shape[0]
        torch_gts = torch.cat((torch_gts,
                               bert_antibody_full_tokens['input_ids'].reshape(-1)), 
                               dim=0)
        out = torch.permute(out, (0, 2, 1))
        torch_preds = torch.cat((torch_preds,
                                 torch.argmax(out, dim=2, keepdim=False).reshape(-1 )), dim=0)
        total_preds_logits = torch.cat((total_preds_logits,
                                        out.reshape(-1, config['FeedForward']['num_classes'])), 
                                        dim=0)
        
        loop.set_description(f"Loss: {iteration_loss.item()}")
        # break
    
    train_accuracy = torch_metrics_accuracy(torch_preds, 
                                            torch_gts, 
                                            ignore_idx=0,
                                            num_classes=config['FeedForward']['num_classes'])
    for k in range(1, config['topk']):
        train_accuracy_top_k = my_topk_accuracy(total_preds_logits,
                                                torch_gts,
                                                topk=k,
                                                ignore_idx=0,
                                                num_classes=config['Bert']['num_classes'])
        wandb.log({f'train_accuracy_top_{k}': train_accuracy_top_k})
        print(f"Top {k} train accuracy: {train_accuracy_top_k}")
    
    train_loss = total_loss / total_count
    wandb.log({'train_accuracy': train_accuracy,
                'train_loss': train_loss})
    return train_accuracy, train_loss, 0.0

def test_forward_bert(model,
                      data_loader,
                      loss_fn,
                      device,
                      config=None):
    '''
    corresponding test function for training forward bert
    '''
    model = model.eval()
    total_loss = 0
    total_count = 0
    torch_gts = torch.zeros((0), dtype=torch.long).to(device)
    torch_preds = torch.zeros((0), dtype=torch.long).to(device)
    total_preds_logits = torch.zeros((0, config['FeedForward']['num_classes']), 
                                     dtype=torch.float).to(device)

    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='green')
    
    for idx, batch in enumerate(loop):
        bert_antibody_full_tokens, \
        bert_antibody_masked_tokens, \
        bert_virus_full_tokens, \
        cross_attention_mask = batch

        bert_antibody_full_tokens['input_ids'] = bert_antibody_full_tokens['input_ids'].to(device)
        bert_antibody_full_tokens['token_type_ids'] = bert_antibody_full_tokens['token_type_ids'].to(device)
        bert_antibody_full_tokens['attention_mask'] = bert_antibody_full_tokens['attention_mask'].to(device)
        bert_antibody_masked_tokens['input_ids'] = bert_antibody_masked_tokens['input_ids'].to(device)
        bert_antibody_masked_tokens['token_type_ids'] = bert_antibody_masked_tokens['token_type_ids'].to(device)
        bert_antibody_masked_tokens['attention_mask'] = bert_antibody_masked_tokens['attention_mask'].to(device)
        bert_virus_full_tokens['input_ids'] = bert_virus_full_tokens['input_ids'].to(device)
        bert_virus_full_tokens['token_type_ids'] = bert_virus_full_tokens['token_type_ids'].to(device)
        bert_virus_full_tokens['attention_mask'] = bert_virus_full_tokens['attention_mask'].to(device)
        cross_attention_mask = cross_attention_mask.to(device)

        # squeeze the second dimension
        bert_antibody_full_tokens['input_ids'] = bert_antibody_full_tokens['input_ids'].squeeze(1)
        bert_antibody_full_tokens['token_type_ids'] = bert_antibody_full_tokens['token_type_ids'].squeeze(1)
        bert_antibody_full_tokens['attention_mask'] = bert_antibody_full_tokens['attention_mask'].squeeze(1)
        bert_antibody_masked_tokens['input_ids'] = bert_antibody_masked_tokens['input_ids'].squeeze(1)
        bert_antibody_masked_tokens['token_type_ids'] = bert_antibody_masked_tokens['token_type_ids'].squeeze(1)
        bert_antibody_masked_tokens['attention_mask'] = bert_antibody_masked_tokens['attention_mask'].squeeze(1)
        bert_virus_full_tokens['input_ids'] = bert_virus_full_tokens['input_ids'].squeeze(1)
        bert_virus_full_tokens['token_type_ids'] = bert_virus_full_tokens['token_type_ids'].squeeze(1)
        bert_virus_full_tokens['attention_mask'] = bert_virus_full_tokens['attention_mask'].squeeze(1)

        with torch.no_grad():
            out = model(bert_antibody_full_tokens,
                        bert_antibody_masked_tokens,
                        bert_virus_full_tokens,
                        cross_attention_mask)
            iteration_loss = loss_fn(out, 
                                    bert_antibody_full_tokens['input_ids'])
        
        total_loss += iteration_loss.item() * bert_antibody_full_tokens['input_ids'].shape[0]
        total_count += bert_antibody_full_tokens['input_ids'].shape[0]
        torch_gts = torch.cat((torch_gts,
                               bert_antibody_full_tokens['input_ids'].reshape(-1)), 
                               dim=0)
        out = torch.permute(out, (0, 2, 1))
        torch_preds = torch.cat((torch_preds,
                                 torch.argmax(out, dim=2, keepdim=False).reshape(-1 )), dim=0)
        total_preds_logits = torch.cat((total_preds_logits,
                                        out.reshape(-1, config['FeedForward']['num_classes'])), 
                                        dim=0)
        loop.set_description(f"Loss: {iteration_loss.item()}")

    test_accuracy = torch_metrics_accuracy(torch_preds,
                                           torch_gts,
                                           ignore_idx=0,
                                           num_classes=config['FeedForward']['num_classes'])
    for k in range(1, config['topk']):
        test_accuracy_top_k = my_topk_accuracy(total_preds_logits,
                                                torch_gts,
                                                topk=k,
                                                ignore_idx=0,
                                                num_classes=config['Bert']['num_classes'])
        wandb.log({f'test_accuracy_top_{k}': test_accuracy_top_k})
        print(f"Top {k} test accuracy: {test_accuracy_top_k}") 
    test_loss = total_loss / total_count
    wandb.log({'test_accuracy': test_accuracy,
               'test_loss': test_loss})
    return test_accuracy, test_loss, 0.0

def train_forward_virus_and_antibody_bert(model,
                                          data_loader,
                                          optimizer,
                                          loss_fn,
                                          device,
                                          config=None):
    '''
    Train the forward virus and antibody bert model.
    Concatenate the antibody and virus sequences and feed them into the model.
    Predict the masked cdr sequence.
    Uses ModelBertAntibodyAndVirus class from model.py.
    '''
    model = model.train()
    total_loss = 0
    total_count = 0
    torch_gts = torch.zeros((0), dtype=torch.long).to(device)
    torch_preds = torch.zeros((0), dtype=torch.long).to(device)
    total_preds_logits = torch.zeros((0, config['Bert']['num_classes']), 
                                     dtype=torch.float).to(device)

    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='red')
    
    for idx, batch in enumerate(loop):
        antibody_and_virus_full_tokens, \
        antibody_and_virus_masked_tokens = batch

        antibody_and_virus_full_tokens['input_ids'] = antibody_and_virus_full_tokens['input_ids'].to(device)
        antibody_and_virus_full_tokens['token_type_ids'] = antibody_and_virus_full_tokens['token_type_ids'].to(device)
        antibody_and_virus_full_tokens['attention_mask'] = antibody_and_virus_full_tokens['attention_mask'].to(device)
        antibody_and_virus_masked_tokens['input_ids'] = antibody_and_virus_masked_tokens['input_ids'].to(device)
        antibody_and_virus_masked_tokens['token_type_ids'] = antibody_and_virus_masked_tokens['token_type_ids'].to(device)
        antibody_and_virus_masked_tokens['attention_mask'] = antibody_and_virus_masked_tokens['attention_mask'].to(device)

        # squeeze the second dimension
        antibody_and_virus_full_tokens['input_ids'] = antibody_and_virus_full_tokens['input_ids'].squeeze(1)
        antibody_and_virus_full_tokens['token_type_ids'] = antibody_and_virus_full_tokens['token_type_ids'].squeeze(1)
        antibody_and_virus_full_tokens['attention_mask'] = antibody_and_virus_full_tokens['attention_mask'].squeeze(1)
        antibody_and_virus_masked_tokens['input_ids'] = antibody_and_virus_masked_tokens['input_ids'].squeeze(1)
        antibody_and_virus_masked_tokens['token_type_ids'] = antibody_and_virus_masked_tokens['token_type_ids'].squeeze(1)
        antibody_and_virus_masked_tokens['attention_mask'] = antibody_and_virus_masked_tokens['attention_mask'].squeeze(1)

        out = model(antibody_and_virus_masked_tokens)
        iteration_loss = loss_fn(out,
                                 antibody_and_virus_full_tokens['input_ids'])
        
        optimizer.zero_grad()
        iteration_loss.backward()
        optimizer.step()

        total_loss += iteration_loss.item() * antibody_and_virus_full_tokens['input_ids'].shape[0]
        total_count += antibody_and_virus_full_tokens['input_ids'].shape[0]
        torch_gts = torch.cat((torch_gts,
                               antibody_and_virus_full_tokens['input_ids'].reshape(-1)),
                              dim=0)
        out = torch.permute(out, (0, 2, 1))
        torch_preds = torch.cat((torch_preds,
                                 torch.argmax(out, dim=2, keepdim=False).reshape(-1 )), dim=0)
        total_preds_logits = torch.cat((total_preds_logits,
                                        out.reshape(-1, config['Bert']['num_classes'])), dim=0)
        loop.set_description(f"Loss: {iteration_loss.item()}")
        # break

    train_accuracy = torch_metrics_accuracy(torch_preds,
                                            torch_gts,
                                            num_classes=config['Bert']['num_classes'],
                                            ignore_idx=0)
    
    for k in range(1, config['topk']):
        # print(torch_gts.shape)
        # print(total_preds_logits.shape)
        train_accuracy_top_k = my_topk_accuracy(total_preds_logits,
                                                torch_gts,
                                                topk=k,
                                                ignore_idx=0,
                                                num_classes=config['Bert']['num_classes'])
        wandb.log({f'train_accuracy_top_{k}': train_accuracy_top_k})
        print(f"Top {k} train accuracy: {train_accuracy_top_k}")
    
    train_loss = total_loss / total_count
    wandb.log({'train_accuracy': train_accuracy,
               'train_loss': train_loss})
    return train_accuracy, train_loss, 0.0

def test_forward_virus_and_antibody_bert(model,
                                         data_loader,
                                         loss_fn,
                                         device,
                                         config=None):
    '''
    Corresponding test function for the forward virus and antibody bert model.
    
    '''
    model = model.eval()
    total_loss = 0
    total_count = 0
    torch_gts = torch.zeros((0), dtype=torch.long).to(device)
    torch_preds = torch.zeros((0), dtype=torch.long).to(device)
    total_preds_logits = torch.zeros((0, config['Bert']['num_classes']),
                                      dtype=torch.float).to(device)

    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='blue')
    
    for idx, batch in enumerate(loop):
        antibody_and_virus_full_tokens, \
        antibody_and_virus_masked_tokens = batch

        antibody_and_virus_full_tokens['input_ids'] = antibody_and_virus_full_tokens['input_ids'].to(device)
        antibody_and_virus_full_tokens['token_type_ids'] = antibody_and_virus_full_tokens['token_type_ids'].to(device)
        antibody_and_virus_full_tokens['attention_mask'] = antibody_and_virus_full_tokens['attention_mask'].to(device)
        antibody_and_virus_masked_tokens['input_ids'] = antibody_and_virus_masked_tokens['input_ids'].to(device)
        antibody_and_virus_masked_tokens['token_type_ids'] = antibody_and_virus_masked_tokens['token_type_ids'].to(device)
        antibody_and_virus_masked_tokens['attention_mask'] = antibody_and_virus_masked_tokens['attention_mask'].to(device)
        
        # squeeze the second dimension
        antibody_and_virus_full_tokens['input_ids'] = antibody_and_virus_full_tokens['input_ids'].squeeze(1)
        antibody_and_virus_full_tokens['token_type_ids'] = antibody_and_virus_full_tokens['token_type_ids'].squeeze(1)
        antibody_and_virus_full_tokens['attention_mask'] = antibody_and_virus_full_tokens['attention_mask'].squeeze(1)
        antibody_and_virus_masked_tokens['input_ids'] = antibody_and_virus_masked_tokens['input_ids'].squeeze(1)
        antibody_and_virus_masked_tokens['token_type_ids'] = antibody_and_virus_masked_tokens['token_type_ids'].squeeze(1)
        antibody_and_virus_masked_tokens['attention_mask'] = antibody_and_virus_masked_tokens['attention_mask'].squeeze(1)

        with torch.no_grad():
            out = model(antibody_and_virus_masked_tokens)
            iteration_loss = loss_fn(out,
                                     antibody_and_virus_full_tokens['input_ids'])

        total_loss += iteration_loss.item() * antibody_and_virus_full_tokens['input_ids'].shape[0]
        total_count += antibody_and_virus_full_tokens['input_ids'].shape[0]
        torch_gts = torch.cat((torch_gts,
                               antibody_and_virus_full_tokens['input_ids'].reshape(-1)),
                               dim=0)
        
        out = torch.permute(out, (0, 2, 1))
        torch_preds = torch.cat((torch_preds,
                                 torch.argmax(out, dim=2, keepdim=False).reshape(-1 )), dim=0)
        total_preds_logits = torch.cat((total_preds_logits,
                                        out.reshape(-1, config['Bert']['num_classes'])), dim=0)
        
        loop.set_description(f"Loss: {iteration_loss.item()}")

    # test_accuracy = torch_metrics_accuracy(torch_preds,
    #                                        torch_gts,
    #                                        ignore_idx=0)
    
    test_accuracy = torch_metrics_accuracy(torch_preds,
                                            torch_gts,
                                            num_classes=config['Bert']['num_classes'],
                                            ignore_idx=0)
    
    for k in range(1, config['topk']):
        test_accuracy_top_k = my_topk_accuracy(total_preds_logits,
                                                torch_gts,
                                                topk=k,
                                                ignore_idx=0,
                                                num_classes=config['Bert']['num_classes'])
        wandb.log({f'test_accuracy_top_{k}': test_accuracy_top_k})
        print(f"Top {k} test accuracy: {test_accuracy_top_k}")

    test_loss = total_loss / total_count
    wandb.log({'test_accuracy': test_accuracy,
                'test_loss': test_loss})
    return test_accuracy, test_loss, 0.0

def run_inference(model, 
                  data_loader, 
                  loss_fn,
                  device,
                  config,
                  bert_mask_token_id,
                  bert_vocab_inv):
    '''
    Run inference on the model
    '''
    model = model.eval()
    total_loss = 0
    total_count = 0
    torch_gts = torch.zeros((0), dtype=torch.long).to(device)
    torch_preds = torch.zeros((0), dtype=torch.long).to(device)
    total_preds_logits = torch.zeros((0, config['Bert']['num_classes']),
                                      dtype=torch.float).to(device)
    
    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='blue')
    
    for idx, batch in enumerate(loop):
        antibody_and_virus_full_tokens, \
        antibody_and_virus_masked_tokens = batch

        antibody_and_virus_full_tokens['input_ids'] = antibody_and_virus_full_tokens['input_ids'].squeeze(1).to(device)
        antibody_and_virus_full_tokens['token_type_ids'] = antibody_and_virus_full_tokens['token_type_ids'].squeeze(1).to(device)
        antibody_and_virus_full_tokens['attention_mask'] = antibody_and_virus_full_tokens['attention_mask'].squeeze(1).to(device)
        antibody_and_virus_masked_tokens['input_ids'] = antibody_and_virus_masked_tokens['input_ids'].squeeze(1).to(device)
        antibody_and_virus_masked_tokens['token_type_ids'] = antibody_and_virus_masked_tokens['token_type_ids'].squeeze(1).to(device)
        antibody_and_virus_masked_tokens['attention_mask'] = antibody_and_virus_masked_tokens['attention_mask'].squeeze(1).to(device)

        with torch.no_grad():
            out = model(antibody_and_virus_masked_tokens)
            iteration_loss = loss_fn(out,
                                     antibody_and_virus_full_tokens['input_ids'])
            
        total_loss += iteration_loss.item() * antibody_and_virus_full_tokens['input_ids'].shape[0]
        total_count += antibody_and_virus_full_tokens['input_ids'].shape[0]
        torch_gts = torch.cat((torch_gts,
                               antibody_and_virus_full_tokens['input_ids'].reshape(-1)),
                               dim=0)
        out = torch.permute(out, (0, 2, 1))
        get_mask_alphabets(antibody_and_virus_masked_tokens['input_ids'],
                           torch.argmax(out, dim=2, keepdim=False),
                           antibody_and_virus_full_tokens['input_ids'],
                           bert_mask_token_id,
                           bert_vocab_inv)
        torch_preds = torch.cat((torch_preds,
                                 torch.argmax(out, dim=2, keepdim=False).reshape(-1 )), dim=0)
        total_preds_logits = torch.cat((total_preds_logits,
                                        out.reshape(-1, config['Bert']['num_classes'])), dim=0)
        
        loop.set_description(f"Loss: {iteration_loss.item()}")

    test_accuracy = torch_metrics_accuracy(torch_preds,
                                            torch_gts,
                                            num_classes=config['Bert']['num_classes'],
                                            ignore_idx=0)
    
    for k in range(1, config['topk']):
        test_accuracy_top_k = my_topk_accuracy(total_preds_logits,
                                               torch_gts,
                                               topk=k,
                                               ignore_idx=0,
                                               num_classes=config['Bert']['num_classes'])
        wandb.log({f'test_accuracy_top_{k}': test_accuracy_top_k})
        print(f"Top {k} test accuracy: {test_accuracy_top_k}")
        
def get_mask_alphabets(input_tensor,
                       pred_tensor,
                       gt_tensor,
                       bert_mask_token_id,
                       bert_vocab_inv):
    input_tensor = input_tensor.reshape(-1)
    pred_tensor = pred_tensor.reshape(-1)
    gt_tensor = gt_tensor.reshape(-1)
    input_mask = ''
    pred_mask = ''
    gt_mask = ''
    for i in range(input_tensor.shape[0]):
        if input_tensor[i] == bert_mask_token_id:
                # input_mask += bert_vocab_inv[int(input_tensor[i])] + ' '
            pred_mask += bert_vocab_inv[int(pred_tensor[i])]
            gt_mask += bert_vocab_inv[int(gt_tensor[i])]
    # print(f"Input mask: {input_mask}")
    print(f"Pred mask: {pred_mask}")
    print(f"gt mask  : {gt_mask}")

def train_forward_autoregressive(model,
                                 data_loader,
                                 optimizer,
                                 loss_fn,
                                 device,
                                 config):
    '''
    Run training on the model in an autoregressive fashion
    This uses Bert Encoder and decoder from torch.nn.Transformer
    Uses AutoRegressiveEncoder from model.py
    '''
    model = model.train()
    total_loss = 0
    total_count = 0
    torch_gts = torch.zeros((0), dtype=torch.long).to(device)
    torch_preds = torch.zeros((0), dtype=torch.long).to(device)
    total_preds_logits = torch.zeros((0, config['Bert']['num_classes']),
                                      dtype=torch.float).to(device)
    
    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='blue')
    
    for idx, batch in enumerate(loop):
        bert_vl_and_virus_full_tokens, \
        bert_cdr_full_tokens = batch

        bert_vl_and_virus_full_tokens['input_ids'] = bert_vl_and_virus_full_tokens['input_ids'].squeeze(1).to(device)
        bert_vl_and_virus_full_tokens['token_type_ids'] = bert_vl_and_virus_full_tokens['token_type_ids'].squeeze(1).to(device)
        bert_vl_and_virus_full_tokens['attention_mask'] = bert_vl_and_virus_full_tokens['attention_mask'].squeeze(1).to(device)
        bert_cdr_full_tokens['input_ids'] = bert_cdr_full_tokens['input_ids'].squeeze(1).to(device)
        bert_cdr_full_tokens['token_type_ids'] = bert_cdr_full_tokens['token_type_ids'].squeeze(1).to(device)
        bert_cdr_full_tokens['attention_mask'] = bert_cdr_full_tokens['attention_mask'].squeeze(1).to(device)

        out = model(bert_vl_and_virus_full_tokens)
        iteration_loss = loss_fn(out,
                                 bert_cdr_full_tokens['input_ids'])
        
        optimizer.zero_grad()
        iteration_loss.backward()
        optimizer.step()

        total_loss += iteration_loss.item() * bert_cdr_full_tokens['input_ids'].shape[0]
        total_count += bert_cdr_full_tokens['input_ids'].shape[0]
        torch_gts = torch.cat((torch_gts,
                               bert_cdr_full_tokens['input_ids'].reshape(-1)),
                                 dim=0)
        out = torch.permute(out, (0, 2, 1))
        torch_preds = torch.cat((torch_preds,
                                 torch.argmax(out, dim=2, keepdim=False).reshape(-1 )), dim=0)
        total_preds_logits = torch.cat((total_preds_logits,
                                        out.reshape(-1, config['Bert']['num_classes'])), dim=0)
        loop.set_description(f"Loss: {iteration_loss.item()}")

    train_accuracy = torch_metrics_accuracy(torch_preds,
                                            torch_gts,
                                            num_classes=config['Bert']['num_classes'],
                                            ignore_idx=0)
    for k in range(1, config['topk']):
        train_accuracy_top_k = my_topk_accuracy(total_preds_logits,
                                                torch_gts,
                                                topk=k,
                                                ignore_idx=0,
                                                num_classes=config['Bert']['num_classes'])
        wandb.log({f'train_accuracy_top_{k}': train_accuracy_top_k})
        print(f"Top {k} train accuracy: {train_accuracy_top_k}")

    return total_loss / total_count, train_accuracy, 0.0

def test_forward_autoregressive(model,
                                data_loader,
                                loss_fn,
                                device,
                                config
                                ):
    '''
    Corresponding test function for train_forward_autoregressive
    '''
    model = model.eval()

    total_loss = 0
    total_count = 0
    torch_gts = torch.zeros((0), dtype=torch.long).to(device)
    torch_preds = torch.zeros((0), dtype=torch.long).to(device)
    total_preds_logits = torch.zeros((0, config['Bert']['num_classes']),
                                      dtype=torch.float).to(device)

    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='blue')
    
    for idx, batch in enumerate(loop):
        bert_vl_and_virus_full_tokens, \
        bert_cdr_full_tokens = batch

        bert_vl_and_virus_full_tokens['input_ids'] = bert_vl_and_virus_full_tokens['input_ids'].squeeze(1).to(device)
        bert_vl_and_virus_full_tokens['token_type_ids'] = bert_vl_and_virus_full_tokens['token_type_ids'].squeeze(1).to(device)
        bert_vl_and_virus_full_tokens['attention_mask'] = bert_vl_and_virus_full_tokens['attention_mask'].squeeze(1).to(device)
        bert_cdr_full_tokens['input_ids'] = bert_cdr_full_tokens['input_ids'].squeeze(1).to(device)
        bert_cdr_full_tokens['token_type_ids'] = bert_cdr_full_tokens['token_type_ids'].squeeze(1).to(device)
        bert_cdr_full_tokens['attention_mask'] = bert_cdr_full_tokens['attention_mask'].squeeze(1).to(device)

        with torch.no_grad():
            out = model(bert_vl_and_virus_full_tokens)
            iteration_loss = loss_fn(out,
                                    bert_cdr_full_tokens['input_ids'])

        total_loss += iteration_loss.item() * bert_cdr_full_tokens['input_ids'].shape[0]
        total_count += bert_cdr_full_tokens['input_ids'].shape[0]
        torch_gts = torch.cat((torch_gts,
                               bert_cdr_full_tokens['input_ids'].reshape(-1)),
                                 dim=0)
        out = torch.permute(out, (0, 2, 1))
        torch_preds = torch.cat((torch_preds,
                                 torch.argmax(out, dim=2, keepdim=False).reshape(-1 )), dim=0)
        total_preds_logits = torch.cat((total_preds_logits,
                                        out.reshape(-1, config['Bert']['num_classes'])), dim=0)
        loop.set_description(f"Loss: {iteration_loss.item()}")

    test_accuracy = torch_metrics_accuracy(torch_preds,
                                           torch_gts,
                                           num_classes=config['Bert']['num_classes'],
                                           ignore_idx=0)
    
    for k in range(1, config['topk']):
        test_accuracy_top_k = my_topk_accuracy(total_preds_logits,
                                               torch_gts,
                                               topk=k,
                                               ignore_idx=0,
                                               num_classes=config['Bert']['num_classes'])
        wandb.log({f'test_accuracy_top_{k}': test_accuracy_top_k})
        print(f"Top {k} test accuracy: {test_accuracy_top_k}")

    return total_loss / total_count, test_accuracy, 0.0

def train_forward_classification(model,
                                 data_loader,
                                 optimizer,
                                 loss_fn,
                                 device,
                                 config):
    '''
    Train function for forward classification
    Used for virus type classification, classes include
    1. SARS-CoV-2
    2. SARS-CoV-WT
    3. SARS-CoV-Omicron etc
    Uses ClassificationModel from model.py
    '''
    num_classes = config['Classification']['class_num']
    model = model.train()
    total_loss = 0
    total_count = 0
    torch_gts = torch.zeros((0, num_classes), dtype=torch.long).to(device)
    torch_preds = torch.zeros((0, num_classes), dtype=torch.long).to(device)

    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='blue')
    
    for idx, batch in enumerate(loop):
        antibodies_full_tokens, \
        labels = batch

        antibodies_full_tokens['input_ids'] = antibodies_full_tokens['input_ids'].squeeze(1).to(device)
        antibodies_full_tokens['token_type_ids'] = antibodies_full_tokens['token_type_ids'].squeeze(1).to(device)
        antibodies_full_tokens['attention_mask'] = antibodies_full_tokens['attention_mask'].squeeze(1).to(device)
        labels = labels.to(device)

        out = model(antibodies_full_tokens)
        iteration_loss = loss_fn(out, labels)

        optimizer.zero_grad()
        iteration_loss.backward()
        optimizer.step()

        total_loss += iteration_loss.item() * antibodies_full_tokens['input_ids'].shape[0]
        total_count += antibodies_full_tokens['input_ids'].shape[0]
        torch_gts = torch.cat((torch_gts,
                               labels.reshape(-1, num_classes)),
                               dim=0)
        preds_sigmoid = torch.sigmoid(out)
        preds_label = torch.zeros_like(preds_sigmoid)
        preds_label[preds_sigmoid >= 0.5] = 1
        torch_preds = torch.cat((torch_preds,
                                 preds_label.reshape(-1, num_classes)), 
                                 dim=0)
        loop.set_description(f"Loss: {iteration_loss.item()}")
        # break

    train_accuracy = multilabel_accuracy(torch_preds,
                                         torch_gts)
    confusion_matrix(torch_preds,
                     torch_gts,
                     num_classes=num_classes)
    
    for class_id in range(num_classes):
        req_preds = torch_preds[:, class_id]
        req_gts = torch_gts[:, class_id]
        train_accuracy_class = multilabel_accuracy(req_preds,
                                                      req_gts)
        wandb.log({f'train_accuracy_class_{class_id}': train_accuracy_class})
        print(f"Class {class_id} train accuracy: {train_accuracy_class}")
    
    return total_loss / total_count, train_accuracy, 0.0
        
def test_forward_classification(model,
                                data_loader,
                                loss_fn,
                                device,
                                config
                                ):
    '''
    Corresponding test function for forward train_forward_classification
    '''
    model = model.eval()
    total_loss = 0
    total_count = 0
    torch_gts = torch.zeros((0, config['Classification']['class_num']), dtype=torch.long).to(device)
    torch_preds = torch.zeros((0, config['Classification']['class_num']), dtype=torch.long).to(device)

    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='green')
    
    for idx, batch in enumerate(loop):
        antibodies_full_tokens, \
        labels = batch

        antibodies_full_tokens['input_ids'] = antibodies_full_tokens['input_ids'].squeeze(1).to(device)
        antibodies_full_tokens['token_type_ids'] = antibodies_full_tokens['token_type_ids'].squeeze(1).to(device)
        antibodies_full_tokens['attention_mask'] = antibodies_full_tokens['attention_mask'].squeeze(1).to(device)
        labels = labels.to(device)

        with torch.no_grad():
            out = model(antibodies_full_tokens)
            iteration_loss = loss_fn(out, labels)

        total_loss += iteration_loss.item() * antibodies_full_tokens['input_ids'].shape[0]
        total_count += antibodies_full_tokens['input_ids'].shape[0]
        torch_gts = torch.cat((torch_gts,
                               labels.reshape(-1, config['Classification']['class_num'])),
                               dim=0)
        preds_sigmoid = torch.sigmoid(out)
        preds_label = torch.zeros_like(preds_sigmoid)
        preds_label[preds_sigmoid >= 0.5] = 1
        torch_preds = torch.cat((torch_preds,
                                 preds_label.reshape(-1, config['Classification']['class_num'])),
                                 dim=0)
        loop.set_description(f"Loss: {iteration_loss.item()}")
        # break

    test_accuracy = multilabel_accuracy(torch_preds,
                                        torch_gts)
    confusion_matrix(torch_preds,
                     torch_gts,
                     num_classes=config['Classification']['class_num'])
    for class_id in range(config['Classification']['class_num']):
        req_preds = torch_preds[:, class_id]
        req_gts = torch_gts[:, class_id]
        test_accuracy_class = multilabel_accuracy(req_preds,
                                                  req_gts)
        wandb.log({f'test_accuracy_class_{class_id}': test_accuracy_class})
        print(f"Class {class_id} test accuracy: {test_accuracy_class}")

    return total_loss / total_count, test_accuracy, 0.0

def train_forward_binding_classification(model,
                                         data_loader,
                                         optimizer,
                                         loss_fn,
                                         device,
                                         config):
    '''
    Train function for forward binding classification to 
    Binds to, No binds to
    Neautralize, No neutralize
    Uses ClassificationModel, TorchTransformerClassificationModel
    '''
    model = model.train()
    total_loss = 0
    total_count = 0
    total_gts = torch.zeros((0, 1), dtype=torch.long).to(device)
    total_preds = torch.zeros((0, 1), dtype=torch.long).to(device)
    total_logits = torch.zeros((0, 2), dtype=torch.float).to(device)

    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='blue')
    
    for idx, batch in enumerate(loop):
        antibodies_full_tokens, \
        labels = batch

        antibodies_full_tokens['input_ids'] = antibodies_full_tokens['input_ids'].squeeze(1).to(device)
        antibodies_full_tokens['token_type_ids'] = antibodies_full_tokens['token_type_ids'].squeeze(1).to(device)
        antibodies_full_tokens['attention_mask'] = antibodies_full_tokens['attention_mask'].squeeze(1).to(device)
        labels = labels.to(device)

        out = model(antibodies_full_tokens)
        iteration_loss = loss_fn(out, labels)

        optimizer.zero_grad()
        iteration_loss.backward()
        optimizer.step()

        total_loss += iteration_loss.item() * antibodies_full_tokens['input_ids'].shape[0]
        total_count += antibodies_full_tokens['input_ids'].shape[0]
        total_gts = torch.cat((total_gts,
                               labels.reshape(-1, 1)),
                               dim=0)
        total_logits = torch.cat((total_logits,
                                  out.reshape(-1, 2)),
                                  dim=0)
        preds = torch.argmax(out, dim=1)
        total_preds = torch.cat((total_preds,
                                 preds.reshape(-1, 1)),
                                 dim=0)
        loop.set_description(f"Loss: {iteration_loss.item()}")
        # break
    total_logits = torch.softmax(total_logits, dim=1)
    train_accuracy = torch_metrics_accuracy(total_preds,
                                            total_gts,
                                            num_classes=2,
                                            ignore_idx=-1)
    thresholds = np.arange(0, 1, 0.05)
    for threshold in thresholds:
        train_biased_accuracy, biased_cm = biased_accuracy(total_logits,
                                                           total_gts,
                                                           threshold=threshold)
        # wandb.log({f'train_biased_accuracy_{threshold}': train_biased_accuracy})
        print(f"Train Biased Accuracy {threshold}: {train_biased_accuracy}")
        print(f"Train Biased Confusion Matrix {threshold}\n: {biased_cm}")

    total_preds_np = total_preds.cpu().numpy()
    total_gts_np = total_gts.cpu().numpy()
    confusion_mat = sk_confusion_matrix(total_gts_np, total_preds_np)
    print(f"Train Confusion Matrix:\n {confusion_mat}")

    return train_accuracy, total_loss / total_count, 0.0

def test_forward_binding_classification(model,
                                        data_loader,
                                        loss_fn,
                                        device,
                                        config):
    '''
    Test function for training forward binding classification
    '''
    model = model.eval()
    total_loss = 0
    total_count = 0
    total_gts = torch.zeros((0, 1), dtype=torch.long).to(device)
    total_preds = torch.zeros((0, 1), dtype=torch.long).to(device)
    total_logits = torch.zeros((0, 2), dtype=torch.float).to(device)

    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='green')

    for idx, batch in enumerate(loop):
        antibodies_full_tokens, \
        labels = batch

        antibodies_full_tokens['input_ids'] = antibodies_full_tokens['input_ids'].squeeze(1).to(device)
        antibodies_full_tokens['token_type_ids'] = antibodies_full_tokens['token_type_ids'].squeeze(1).to(device)
        antibodies_full_tokens['attention_mask'] = antibodies_full_tokens['attention_mask'].squeeze(1).to(device)
        labels = labels.to(device)

        with torch.no_grad():
            out = model(antibodies_full_tokens)
            iteration_loss = loss_fn(out, labels)

        total_loss += iteration_loss.item() * antibodies_full_tokens['input_ids'].shape[0]
        total_count += antibodies_full_tokens['input_ids'].shape[0]
        total_gts = torch.cat((total_gts,
                               labels.reshape(-1, 1)),
                               dim=0)
        total_logits = torch.cat((total_logits,
                                    out.reshape(-1, 2)),
                                    dim=0)
        preds = torch.argmax(out, dim=1)
        total_preds = torch.cat((total_preds,
                                 preds.reshape(-1, 1)),
                                 dim=0)
        loop.set_description(f"Loss: {iteration_loss.item()}")

    total_logits = torch.softmax(total_logits, dim=1)
    test_accuracy = torch_metrics_accuracy(total_preds,
                                           total_gts,
                                           num_classes=2,
                                           ignore_idx=-1)
    thresholds = np.arange(0, 1, 0.05)
    for threshold in thresholds:
        test_biased_accuracy, biased_cm = biased_accuracy(total_logits,
                                                          total_gts,
                                                          threshold=threshold)
        # wandb.log({f'test_biased_accuracy_{threshold}': test_biased_accuracy})
        print(f"Test Biased Accuracy {threshold}: {test_biased_accuracy}")
        print(f"Test Biased Confusion Matrix {threshold}\n: {biased_cm}")

    total_preds_np = total_preds.cpu().numpy()
    total_gts_np = total_gts.cpu().numpy()
    confusion_mat = sk_confusion_matrix(total_gts_np, total_preds_np)
    print("Test Confusion Matrix\n", confusion_mat)

    return test_accuracy, total_loss / total_count, 0.0                              

def train_forward_classification_with_decoder(model,
                                              data_loader,
                                              optimizer,
                                              loss_fn,
                                              device,
                                              config):
    '''
    Train function for training forward classification with decoder
    Encodes virus and antibody and then decodes, with antibody as the decoder input
    and virus as key and value
    Uses ClassificationDecoderModel from model.py
    '''
    model = model.train()
    total_loss = 0
    total_count = 0
    total_gts = torch.zeros((0, 1), dtype=torch.long).to(device)
    total_preds = torch.zeros((0, 1), dtype=torch.long).to(device)
    total_logits = torch.zeros((0, 2), dtype=torch.float).to(device)
    
    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='blue')
    
    for idx, batch in enumerate(loop):
        antibody_tokens, \
        virus_tokens, \
        labels = batch

        antibody_tokens['input_ids'] = antibody_tokens['input_ids'].squeeze(1).to(device)
        antibody_tokens['token_type_ids'] = antibody_tokens['token_type_ids'].squeeze(1).to(device)
        antibody_tokens['attention_mask'] = antibody_tokens['attention_mask'].squeeze(1).to(device)
        virus_tokens['input_ids'] = virus_tokens['input_ids'].squeeze(1).to(device)
        virus_tokens['token_type_ids'] = virus_tokens['token_type_ids'].squeeze(1).to(device)
        virus_tokens['attention_mask'] = virus_tokens['attention_mask'].squeeze(1).to(device)
        labels = labels.to(device)

        out = model(antibody_tokens, virus_tokens)
        iteration_loss = loss_fn(out, labels)

        optimizer.zero_grad()
        iteration_loss.backward()
        optimizer.step()

        total_loss += iteration_loss.item() * antibody_tokens['input_ids'].shape[0]
        total_count += antibody_tokens['input_ids'].shape[0]
        total_gts = torch.cat((total_gts,
                               labels.reshape(-1, 1)),
                               dim=0)
        total_logits = torch.cat((total_logits,
                                  out.reshape(-1, 2)),
                                    dim=0)
        preds = torch.argmax(out, dim=1)
        total_preds = torch.cat((total_preds,
                                 preds.reshape(-1, 1)),
                                 dim=0)
        loop.set_description(f"Loss: {iteration_loss.item()}")
        # break

    total_logits = torch.softmax(total_logits, dim=1)
    for threshold in [0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98]:    
        train_biased_accuracy, biased_confusion_mat = biased_accuracy(total_logits,
                                                total_gts,
                                                threshold=threshold)
        print(f"Train Biased Accuracy with threshold {threshold}: {train_biased_accuracy}")
        print(f"Train Biased Confusion Matrix with threshold {threshold}:\n {biased_confusion_mat}")

    train_accuracy = torch_metrics_accuracy(total_preds,
                                            total_gts,
                                            num_classes=config['Classification']['class_num'],
                                            ignore_idx=-1)
    total_preds_np = total_preds.cpu().numpy()
    total_gts_np = total_gts.cpu().numpy()
    confusion_mat = sk_confusion_matrix(total_gts_np, total_preds_np)
    print(f"Train Confusion Matrix:\n {confusion_mat}")
    # print(f"Train Biased Accuracy: {train_biased_accuracy}")
    return train_accuracy, total_loss / total_count, 0.0

def test_forward_classification_with_decoder(model,
                                             data_loader,
                                             loss_fn,
                                             device,
                                             config):
    '''
    Test function for train_forward_classification_with_decoder
    '''
    model = model.eval()
    total_loss = 0
    total_count = 0
    total_gts = torch.zeros((0, 1), dtype=torch.long).to(device)
    total_preds = torch.zeros((0, 1), dtype=torch.long).to(device)
    total_logits = torch.zeros((0, 2), dtype=torch.float).to(device)

    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='green')
    
    for idx, batch in enumerate(loop):
        antibody_tokens, \
        virus_tokens, \
        labels = batch

        antibody_tokens['input_ids'] = antibody_tokens['input_ids'].squeeze(1).to(device)
        antibody_tokens['token_type_ids'] = antibody_tokens['token_type_ids'].squeeze(1).to(device)
        antibody_tokens['attention_mask'] = antibody_tokens['attention_mask'].squeeze(1).to(device)
        virus_tokens['input_ids'] = virus_tokens['input_ids'].squeeze(1).to(device)
        virus_tokens['token_type_ids'] = virus_tokens['token_type_ids'].squeeze(1).to(device)
        virus_tokens['attention_mask'] = virus_tokens['attention_mask'].squeeze(1).to(device)
        labels = labels.to(device)

        with torch.no_grad():
            out = model(antibody_tokens, virus_tokens)
            iteration_loss = loss_fn(out, labels)

        total_loss += iteration_loss.item() * antibody_tokens['input_ids'].shape[0]
        total_count += antibody_tokens['input_ids'].shape[0]
        total_gts = torch.cat((total_gts,
                               labels.reshape(-1, 1)),
                               dim=0)
        total_logits = torch.cat((total_logits,
                                  out.reshape(-1, 2)),
                                    dim=0)
        preds = torch.argmax(out, dim=1)
        total_preds = torch.cat((total_preds,
                                 preds.reshape(-1, 1)),
                                 dim=0)
        loop.set_description(f"Loss: {iteration_loss.item()}")
        # break

    total_logits = torch.softmax(total_logits, dim=1)

    test_accuracy = torch_metrics_accuracy(total_preds,
                                           total_gts,
                                           num_classes=config['Classification']['class_num'],
                                             ignore_idx=-1)
    total_preds_np = total_preds.cpu().numpy()
    total_gts_np = total_gts.cpu().numpy()
    confusion_mat = sk_confusion_matrix(total_gts_np, total_preds_np)

    for threshold in [0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98]:
        test_biased_accuracy, test_confusion_mat = biased_accuracy(total_logits,
                                               total_gts,
                                               threshold=threshold)
        print(f"Test Biased Accuracy (threshold={threshold}): {test_biased_accuracy}")
        print(f"Test Biased Confusion Matrix (threshold={threshold}):\n {test_confusion_mat}")
    print(f"Test Confusion Matrix:\n {confusion_mat}")
    return test_accuracy, total_loss / total_count, 0.0

def train_forward_classification_graph(model,
                                       data_loader,
                                       optimizer,
                                       loss_fn,
                                       device,
                                       config):
    '''
    Train function for forward classification with graph which also
    uses graph features from Rdkit
    Uses TransformerWithGraphFeats from model.py
    '''
    model = model.train()
    total_loss = 0
    total_count = 0
    total_gts = torch.zeros((0, 1), dtype=torch.long).to(device)
    total_preds = torch.zeros((0, 1), dtype=torch.long).to(device)
    total_logits = torch.zeros((0, 2), dtype=torch.float).to(device)

    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='green')
    
    for idx, batch in enumerate(loop):
        cdr3_features, \
        virus_features, \
        cdr3_mask, \
        virus_mask, \
        labels = batch

        cdr3_features = cdr3_features.to(device)
        virus_features = virus_features.to(device)
        cdr3_mask = cdr3_mask.to(device)
        virus_mask = virus_mask.to(device)
        labels = labels.to(device)

        out = model(cdr3_features,
                    virus_features,
                    cdr3_mask,
                    virus_mask)
        
        iteration_loss = loss_fn(out, labels)

        optimizer.zero_grad()
        iteration_loss.backward()
        optimizer.step()

        total_loss += iteration_loss.item() * cdr3_features.shape[0]
        total_count += cdr3_features.shape[0]
        total_gts = torch.cat((total_gts,
                               labels.reshape(-1, 1)),
                               dim=0)
        total_logits = torch.cat((total_logits,
                                  out.reshape(-1, 2)),
                                    dim=0)

        preds = torch.argmax(out, dim=1)
        total_preds = torch.cat((total_preds,
                                 preds.reshape(-1, 1)),
                                 dim=0)
        loop.set_description(f"Loss: {iteration_loss.item()}")

    total_logits = torch.softmax(total_logits, dim=1)
    for threshold in [0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98]:
        train_biased_accuracy, train_confusion_mat = biased_accuracy(total_logits,
                                               total_gts,
                                               threshold=threshold)
        print(f"Train Biased Accuracy (threshold={threshold}): {train_biased_accuracy}")
        print(f"Train Biased Confusion Matrix (threshold={threshold}):\n {train_confusion_mat}")
    
    train_accuracy = torch_metrics_accuracy(total_preds,
                                            total_gts,
                                            num_classes=config['Classification']['class_num'],
                                            ignore_idx=-1)
    total_preds_np = total_preds.cpu().numpy()
    total_gts_np = total_gts.cpu().numpy()
    confusion_mat = sk_confusion_matrix(total_gts_np, total_preds_np)
    print(f"Train Confusion Matrix:\n {confusion_mat}")
    return train_accuracy, total_loss / total_count, 0.0

def test_forward_classification_graph(model,
                                      data_loader,
                                      loss_fn,
                                      device,
                                      config):
    '''
    Corresponding test function for forward classification with graph
    '''
    model = model.eval()
    total_loss = 0
    total_count = 0
    total_gts = torch.zeros((0, 1), dtype=torch.long).to(device)
    total_preds = torch.zeros((0, 1), dtype=torch.long).to(device)
    total_logits = torch.zeros((0, 2), dtype=torch.float).to(device)

    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='green')
    
    for idx, batch in enumerate(loop):
        cdr3_features, \
        virus_features, \
        cdr3_mask, \
        virus_mask, \
        labels = batch

        cdr3_features = cdr3_features.to(device)
        virus_features = virus_features.to(device)
        cdr3_mask = cdr3_mask.to(device)
        virus_mask = virus_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            out = model(cdr3_features,
                        virus_features,
                        cdr3_mask,
                        virus_mask)
            iteration_loss = loss_fn(out, labels)

        total_loss += iteration_loss.item() * cdr3_features.shape[0]
        total_count += cdr3_features.shape[0]
        total_gts = torch.cat((total_gts,
                               labels.reshape(-1, 1)),
                               dim=0)
        total_logits = torch.cat((total_logits,
                                  out.reshape(-1, 2)),
                                    dim=0)
        preds = torch.argmax(out, dim=1)
        total_preds = torch.cat((total_preds,
                                 preds.reshape(-1, 1)),
                                 dim=0)
        loop.set_description(f"Loss: {iteration_loss.item()}")
        # break
    
    total_logits = torch.softmax(total_logits, dim=1)
    for threshold in [0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98]:
        test_biased_accuracy, test_confusion_mat = biased_accuracy(total_logits,
                                               total_gts,
                                               threshold=threshold)
        print(f"Test Biased Accuracy (threshold={threshold}): {test_biased_accuracy}")
        print(f"Test Biased Confusion Matrix (threshold={threshold}):\n {test_confusion_mat}")

    test_accuracy = torch_metrics_accuracy(total_preds,
                                             total_gts,
                                                num_classes=config['Classification']['class_num'],
                                                 ignore_idx=-1)
    total_preds_np = total_preds.cpu().numpy()
    total_gts_np = total_gts.cpu().numpy()
    confusion_mat = sk_confusion_matrix(total_gts_np, total_preds_np)
    print(f"Test Confusion Matrix:\n {confusion_mat}")
    return test_accuracy, total_loss / total_count, 0.0

def train_forward_autoregressive_encoder(model,
                                         data_loader,
                                         optimizer,
                                         loss_fn,
                                         device,
                                         config):
    '''
    Train function for forward autoregressive encoder
    Uses AutoregressiveEncoder from model.py
    '''
    model = model.train()
    total_loss = 0
    total_count = 0
    total_gts = torch.zeros((0, 1), dtype=torch.long).to(device)
    total_preds = torch.zeros((0, 1), dtype=torch.long).to(device)
    total_logits = torch.zeros((0, config['Bert']['num_classes']), dtype=torch.float).to(device)

    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='green')
    
    for idx, batch in enumerate(loop):
        vhh_and_virus_tokenized = batch['vhh_and_virus_tokenized']
        vhh_full_tokenized = batch['vhh_full_tokenized']
        mask_start_idx = batch['mask_start_idx']
        mask_end_idx = batch['mask_end_idx']
        mask_len = batch['mask_len']

        vhh_and_virus_tokenized['input_ids'] = vhh_and_virus_tokenized['input_ids'].to(device).squeeze(1)
        vhh_and_virus_tokenized['token_type_ids'] = vhh_and_virus_tokenized['token_type_ids'].to(device).squeeze(1)
        vhh_and_virus_tokenized['attention_mask'] = vhh_and_virus_tokenized['attention_mask'].to(device).squeeze(1)

        vhh_full_tokenized['input_ids'] = vhh_full_tokenized['input_ids'].to(device).squeeze(1)
        vhh_full_tokenized['token_type_ids'] = vhh_full_tokenized['token_type_ids'].to(device).squeeze(1)
        vhh_full_tokenized['attention_mask'] = vhh_full_tokenized['attention_mask'].to(device).squeeze(1)

        mask_start_idx = mask_start_idx.to(device)
        mask_end_idx = mask_end_idx.to(device)

        out = model(vhh_and_virus_tokenized,
                    vhh_full_tokenized,
                    mask_start_idx,
                    mask_end_idx,
                    is_train=True)
        iteration_loss = loss_fn(out, vhh_full_tokenized['input_ids'])

        optimizer.zero_grad()
        iteration_loss.backward()
        optimizer.step()

        total_loss += iteration_loss.item() * vhh_and_virus_tokenized['input_ids'].shape[0]
        total_count += vhh_and_virus_tokenized['input_ids'].shape[0]

        total_gts = torch.cat((total_gts,
                               vhh_full_tokenized['input_ids'].reshape(-1, 1)),
                               dim=0)
        out = out.permute(0, 2, 1) # (batch_size, seq_len, vocab_size)
        total_preds = torch.cat((total_preds,
                                 torch.argmax(out, dim=2).reshape(-1, 1)),
                                 dim=0)
        total_logits = torch.cat((total_logits,
                                  out.reshape(-1, config['Bert']['num_classes'])),
                                  dim=0)
        loop.set_description(f"Loss: {iteration_loss.item()}")
        # break
                                 
    train_accuracy = torch_metrics_accuracy(total_preds.reshape(-1),
                                            total_gts.reshape(-1),
                                            num_classes=config['Bert']['num_classes'],
                                            ignore_idx=0)
    
    for k in range(1, config['topk']):
        train_topk_accuracy = my_topk_accuracy(total_logits,
                                               total_gts.reshape(-1),
                                               topk=k,
                                               ignore_idx=0,
                                               num_classes=config['Bert']['num_classes'])
        print(f"Train Top-{k} Accuracy: {train_topk_accuracy}")
        wandb.log({f"Train Top-{k} Accuracy": train_topk_accuracy})

    return train_accuracy, total_loss / total_count, 0.0

def test_forward_autoregressive_encoder(model,
                                        data_loader,
                                        loss_fn,
                                        device,
                                        config):
    '''
    Corresponding test function for forward autoregressive encoder
    '''
    model = model.eval()
    total_loss = 0
    total_count = 0
    total_gts = torch.zeros((0, 1), dtype=torch.long).to(device)
    total_preds = torch.zeros((0, 1), dtype=torch.long).to(device)
    total_logits = torch.zeros((0, config['ClassificationTwoPosition']['FeedForward']['num_classes']), dtype=torch.float).to(device)
                            
    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='green')
    
    for idx, batch in enumerate(loop):
        vhh_and_virus_tokenized = batch['vhh_and_virus_tokenized']
        vhh_full_tokenized = batch['vhh_full_tokenized']
        mask_start_idx = batch['mask_start_idx']
        mask_end_idx = batch['mask_end_idx']
        mask_len = batch['mask_len']

        vhh_and_virus_tokenized['input_ids'] = vhh_and_virus_tokenized['input_ids'].to(device).squeeze(1)
        vhh_and_virus_tokenized['token_type_ids'] = vhh_and_virus_tokenized['token_type_ids'].to(device).squeeze(1)
        vhh_and_virus_tokenized['attention_mask'] = vhh_and_virus_tokenized['attention_mask'].to(device).squeeze(1)

        vhh_full_tokenized['input_ids'] = vhh_full_tokenized['input_ids'].to(device).squeeze(1)
        vhh_full_tokenized['token_type_ids'] = vhh_full_tokenized['token_type_ids'].to(device).squeeze(1)
        vhh_full_tokenized['attention_mask'] = vhh_full_tokenized['attention_mask'].to(device).squeeze(1)
        
        mask_start_idx = mask_start_idx.to(device)
        mask_end_idx = mask_end_idx.to(device)
        
        with torch.no_grad():
            out = model(vhh_and_virus_tokenized,
                        vhh_full_tokenized,
                        mask_start_idx,
                        mask_end_idx,
                        is_train=False)
            iteration_loss = loss_fn(out, vhh_full_tokenized['input_ids'])

        total_loss += iteration_loss.item() * vhh_and_virus_tokenized['input_ids'].shape[0]
        total_count += vhh_and_virus_tokenized['input_ids'].shape[0]

        total_gts = torch.cat((total_gts,
                               vhh_full_tokenized['input_ids'].reshape(-1, 1)),
                               dim=0)
        out = out.permute(0, 2, 1)
        total_preds = torch.cat((total_preds,
                                 torch.argmax(out, dim=2).reshape(-1, 1)),
                                 dim=0)
        total_logits = torch.cat((total_logits,
                                  out.reshape(-1, config['Bert']['num_classes'])),
                                  dim=0)
        loop.set_description(f"Loss: {iteration_loss.item()}")
        # break
    test_accuracy = torch_metrics_accuracy(total_preds.reshape(-1),
                                           total_gts.reshape(-1),
                                           num_classes=config['Bert']['num_classes'],
                                           ignore_idx=0)
    
    for k in range(1, config['topk']):
        test_topk_accuracy = my_topk_accuracy(total_logits,
                                              total_gts.reshape(-1),
                                              topk=k,
                                              ignore_idx=0,
                                              num_classes=config['Bert']['num_classes'])
        print(f"Test Top-{k} Accuracy: {test_topk_accuracy}")
        wandb.log({f"Test Top-{k} Accuracy": test_topk_accuracy})

    return test_accuracy, total_loss / total_count, 0.0

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
        wandb.log({f"Train Top-{k} Accuracy": train_topk_accuracy})

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
        wandb.log({f"Test Top-{k} Accuracy": test_topk_accuracy})
    
    return test_accuracy, total_loss / total_count, 0.0
        

def train_forward_ablang(model,
                         data_loader,
                         optimizer,
                         loss_fn,
                         device,
                         config=None):
    '''
    Corresponding train function for forward to predict ablang
    '''
    model = model.train()
    total_loss = 0
    total_count = 0
    total_gts = torch.zeros((0), dtype=torch.long).to(device)
    total_preds = torch.zeros((0), dtype=torch.long).to(device)
    total_logits = torch.zeros((0, config['AbLang']['Classifier']['num_classes']),
                                dtype=torch.float).to(device)
    
    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='green')
    
    for idx, batch in enumerate(loop):
        antibody_masked, \
        antobody_label = batch

        antibody_masked['input_ids'] = antibody_masked['input_ids'].to(device).squeeze(1)
        antibody_masked['token_type_ids'] = antibody_masked['token_type_ids'].to(device).squeeze(1)
        antibody_masked['attention_mask'] = antibody_masked['attention_mask'].to(device).squeeze(1)
        antobody_label['input_ids'] = antobody_label['input_ids'].to(device).squeeze(1)
        antobody_label['token_type_ids'] = antobody_label['token_type_ids'].to(device).squeeze(1)
        antobody_label['attention_mask'] = antobody_label['attention_mask'].to(device).squeeze(1)

        out = model(antibody_masked)
        iteration_loss = loss_fn(out, antobody_label['input_ids'])
        optimizer.zero_grad()
        iteration_loss.backward()
        optimizer.step()

        total_loss += iteration_loss.item() * antibody_masked['input_ids'].shape[0]
        total_count += antibody_masked['input_ids'].shape[0]
        out = out.permute(0, 2, 1) # [batch_size,  seq_len, num_classes]
        total_gts = torch.cat((total_gts,
                               antobody_label['input_ids'].reshape(-1)),
                               dim=0)
        total_logits = torch.cat((total_logits,
                                  out.reshape(-1, config['AbLang']['Classifier']['num_classes'])),
                                  dim=0)
        total_preds = torch.cat((total_preds,
                                 torch.argmax(out, dim=2).reshape(-1)),
                                 dim=0)
        loop.set_description(f"Loss: {iteration_loss.item()}")

    train_accuracy = torch_metrics_accuracy(total_preds.reshape(-1),
                                            total_gts.reshape(-1),
                                            ignore_idx=0,
                                            num_classes=config['AbLang']['Classifier']['num_classes'])
    for k in range(1, config['topk']):
        train_topk_accuracy = my_topk_accuracy(total_logits,
                                               total_gts.reshape(-1),
                                               topk=k,
                                               ignore_idx=0,
                                               num_classes=config['AbLang']['Classifier']['num_classes'])
        print(f"Train Top-{k} Accuracy: {train_topk_accuracy}")
        wandb.log({f"Train Top-{k} Accuracy": train_topk_accuracy})
    
    wandb.log({"Train Accuracy": train_accuracy})
    return train_accuracy, total_loss / total_count, 0.0

def test_forward_ablang(model,
                        data_loader,
                        loss_fn,
                        device,
                        config=None):
    torch.cuda.empty_cache()
    model = model.eval()
    total_loss = 0
    total_count = 0
    total_gts = torch.zeros((0), dtype=torch.long).to(device)
    total_preds = torch.zeros((0), dtype=torch.long).to(device)
    total_logits = torch.zeros((0, config['AbLang']['Classifier']['num_classes']),
                                dtype=torch.float).to(device)
    
    loop = tqdm(data_loader,
                leave=True,
                total=len(data_loader),
                colour='green')
    
    for idx, batch in enumerate(loop):
        antibody_masked, \
        antobody_label = batch

        antibody_masked['input_ids'] = antibody_masked['input_ids'].to(device).squeeze(1)
        antibody_masked['token_type_ids'] = antibody_masked['token_type_ids'].to(device).squeeze(1)
        antibody_masked['attention_mask'] = antibody_masked['attention_mask'].to(device).squeeze(1)

        antobody_label['input_ids'] = antobody_label['input_ids'].to(device).squeeze(1)
        antobody_label['token_type_ids'] = antobody_label['token_type_ids'].to(device).squeeze(1)
        antobody_label['attention_mask'] = antobody_label['attention_mask'].to(device).squeeze(1)

        with torch.no_grad():
            out = model(antibody_masked)
            iteration_loss = loss_fn(out, antobody_label['input_ids'])
        
        total_loss += iteration_loss.item() * antibody_masked['input_ids'].shape[0]
        total_count += antibody_masked['input_ids'].shape[0]
        out = out.permute(0, 2, 1) # [batch_size,  seq_len, num_classes]
        total_gts = torch.cat((total_gts,
                                 antobody_label['input_ids'].reshape(-1)),
                                 dim=0)
        total_logits = torch.cat((total_logits,
                                  out.reshape(-1, config['AbLang']['Classifier']['num_classes'])),
                                  dim=0)
        total_preds = torch.cat((total_preds,
                                 torch.argmax(out, dim=2).reshape(-1)),
                                 dim=0)
        loop.set_description(f"Loss: {iteration_loss.item()}")

    test_accuracy = torch_metrics_accuracy(total_preds.reshape(-1),
                                           total_gts.reshape(-1),
                                             ignore_idx=0,
                                           num_classes=config['AbLang']['Classifier']['num_classes'])
    for k in range(1, config['topk']):
        test_topk_accuracy = my_topk_accuracy(total_logits,
                                              total_gts.reshape(-1),
                                              topk=k,
                                              ignore_idx=0,
                                              num_classes=config['AbLang']['Classifier']['num_classes'])
        print(f"Test Top-{k} Accuracy: {test_topk_accuracy}")
        wandb.log({f"Test Top-{k} Accuracy": test_topk_accuracy})
    
    wandb.log({"Test Accuracy": test_accuracy})
    return test_accuracy, total_loss / total_count, 0.0


if __name__ == "__main__":
    pass
