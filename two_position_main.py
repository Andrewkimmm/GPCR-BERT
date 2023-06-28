import torch
import torch.nn as nn
from colorama import init
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
import yaml
import json
import tqdm
from tqdm import tqdm
from model import ClassificationTwoPositions, ClassificationTwoPositionsBert
from dataset import TwoPositionPredictionDataset, PositionPredictionFullDataset
import wandb
from train import train_forward_two_position_prediction, \
                  test_forward_two_position_prediction

from utils import calculate_num_data_per_class, \
                  confusion_matrix, \
                  make_colored_string, \
                  visualize_attention_weights
import json
#from bertviz import head_view

# function to run inference on test data
def test(model, 
         test_loader, 
         device, 
         config,
         vocab,
         data_name):
    model.eval()
    total_loss = 0
    total_count = 0
    total_gts = torch.zeros((0), dtype=torch.long).to(device)
    total_preds = torch.zeros((0), dtype=torch.long).to(device)
    total_logits = torch.zeros((0, 
                                config['ClassificationTwoPosition']['FeedForward']['num_classes']),
                                dtype=torch.float).to(device)
    # test_loader = PositionPredictionFullDataset
    loop = tqdm(test_loader, 
                total=len(test_loader),
                leave=True) 
    
    # req_idxs = np.random.randint(0, 
    #                              len(test_loader), 
    #                              config['ClassificationTwoPosition']['plot_num'],
    #                              replace=False)
    weight_and_virus = []
    for idx, batch in enumerate(loop):
        input_tokens, \
        label_tokens, \
        start_idx,\
        pad_start = batch\
        
        print(input_tokens)
        print(label_tokens)
        print(start_idx)
        print(pad_start)
        #print(typep(org_str))
        #print(org_str)
        #start_idx,\
        #pad_start,org_str >> start, end index
        #org_str = org_str1[0]
        input_tokens['input_ids'] = input_tokens['input_ids'].to(device).squeeze(1)
        input_tokens['attention_mask'] = input_tokens['attention_mask'].to(device).squeeze(1)
        input_tokens['token_type_ids'] = input_tokens['token_type_ids'].to(device).squeeze(1)
        
        label_tokens['input_ids'] = label_tokens['input_ids'].to(device).squeeze(1)
        label_tokens['attention_mask'] = label_tokens['attention_mask'].to(device).squeeze(1)
        label_tokens['token_type_ids'] = label_tokens['token_type_ids'].to(device).squeeze(1)

        with torch.no_grad():
            if config['ClassificationTwoPosition']['use_bert']:
                out, attention = model.forward_test(input_tokens) 
                # attention: (batch_size, num_heads, seq_len, seq_len)
                # input_ids = input_tokens["input_ids"].squeeze(0).tolist()

                # input_tokens_str = test_loader.dataset.tokenizer.convert_ids_to_tokens(input_ids)
            
                # label_tokens_str = test_loader.dataset.tokenizer.convert_ids_to_tokens()
                # print(input_tokens['input_ids'].shape)
                # input_tokens_str = [vocab[int(x)] for x in input_tokens['input_ids'][0].squeeze(0).tolist()]
                # print(input_tokens_str)
                # print(label_tokens_str)
                # exit()
                # getting attention for the first element in the batch
                # print(attention.shape)

                # head_view(attention, input_tokens_str)
                # exit()
                # print(attention.shape) # (batch_size, num_heads, seq_len, seq_len)
        
        for b_no, (input_token, label_token) in enumerate(zip(input_tokens['input_ids'], 
                                                              label_tokens['input_ids'])):
            input_str_list = [vocab[int(x)] for x in input_token.tolist()]
        
            input_str = "".join(input_str_list)
            #print("input_str = ", input_str)
            label_str_list = [vocab[int(x)] for x in label_token.tolist()]
            #label_str_list = [vocab.get(int(x),"") for x in label_token.tolist()]
            label_str = "".join(label_str_list)
            input_str = input_str.replace('[CLS]', 'J')
            input_str = input_str.replace('[SEP]', 'J')
            input_str = input_str.replace('[PAD]', 'J')
            input_str = input_str.replace('[UNK]', 'J')
            input_str = input_str.replace('[MASK]', 'J')

            label_str = label_str.replace('[CLS]', 'J')
            label_str = label_str.replace('[SEP]', 'J')
            label_str = label_str.replace('[PAD]', 'J')
            label_str = label_str.replace('[UNK]', 'J')
            label_str = label_str.replace('[MASK]', 'J')

            # input_str = make_colored_string(input_str,
            #                                 [int(start_idx[b_no]), int(start_idx[b_no]+1)])
            # label_str = make_colored_string(label_str,
            #                                 [int(start_idx[b_no]), int(start_idx[b_no]+1)])
            curr_idx = idx*input_tokens['input_ids'].shape[0] + b_no
            # print(f"{curr_idx}:, {len(attention)}")
            #print(f"org_str: {org_str[b_no]}")
            print(f"Input: {input_str[1:pad_start[b_no].item()]}")
            req_attention = attention[config['ClassificationTwoPosition']['Bert']['layer_no_to_view']]
            req_attention = req_attention[b_no, 
                                          config['ClassificationTwoPosition']['Bert']['head_no_to_view'],
                                          :,
                                          :].squeeze(0).cpu().numpy()
            # req_attention = req_attention
            # print(req_attention.shape)            
            req_pad_start = pad_start[b_no].item()
            req_attention = req_attention[1:req_pad_start, 1:req_pad_start]
            #req_attention = req_attention[1:len(org_str[b_no])+1, 1:len(org_str[b_no])+1]
            print(req_attention.shape)
            print(len(input_str[1:pad_start[b_no].item()]))
            #print(len(org_str[b_no]))
            weight_and_virus.append({#"pdb" : pdb_name[0],
                                    "seq": input_str[1:pad_start[b_no].item()],#org_str[b_no], 
                                    "attention": req_attention})
            # if curr_idx in req_idxs:
            #     visualize_attention_weights(req_attention,
            #                                 input_str[:req_pad_start],
            #                                 str(curr_idx))

    
    weight_and_virus_np = np.array(weight_and_virus)
    np.save(f"{data_name}_weight_and_sequences.npy", weight_and_virus_np)
    print(weight_and_virus_np.shape)
            # np.save("mattr.npy", matr)
        #     # input_str = ",".join(input_str)
        #     print(f"Input: {input_str}")
        #     print(f"Label: {label_str}")
            # print("="*200)

        # input_str = [vocab[x] for x in input_tokens['input_ids'].squeeze(0).tolist()]
        # input_str = ",".join(input_str)
        # label_str = [vocab[x] for x in label_tokens['input_ids'].squeeze(0).tolist()]
        # label_str = ",".join(label_str)
        # print(f"Input: {input_str}")
        # print(f"Label: {label_str}")
        # print(input_tokens['input_ids'])
        # print(label_tokens['input_ids'])
        # print("="*200)
'''
def custom_collate(batch):
    batch = [item for item in batch if item.size() == batch[0].size()]
    if len(batch) > 0:
        return torch.stack(batch)
    else:
        return None
'''
    #np.savez("attention.npz", c)
def main():
    config_file = 'config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # assign wandb config to config
        # config = {**config, **wandb.config}

    # Load data
    data = np.load(config['ClassificationTwoPosition']['filename'], allow_pickle=True)
    req_save_name = config['ClassificationTwoPosition']['filename'].split("/")[-1].split(".")[0]

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Split data into train and test
    train_data, test_data = train_test_split(data, 
                                             test_size=0.25, 
                                             random_state=41)
    
    # Create dataset
    #train_dataset = TwoPositionPredictionDataset(train_data, config)
    #test_dataset = TwoPositionPredictionDataset(test_data, config)
    train_dataset = PositionPredictionFullDataset(train_data, config)
    test_dataset = PositionPredictionFullDataset(test_data, config)
    
    #train_dataset = TwoPositionPredictionDataset(train_data, config)
    #test_dataset = TwoPositionPredictionDataset(test_data, config)


    # bert_tokenizer
    bert_vocab = train_dataset.tokenizer.vocab

    # Create dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config['ClassificationTwoPosition']['batch_size'],
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config['ClassificationTwoPosition']['batch_size'],
                                              shuffle=False)

    # Create model
    # model = ClassificationTwoPositions(config).to(device)
    model = ClassificationTwoPositionsBert(config).to(device)

    # Create optimizer, loss function, scheduler
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['ClassificationTwoPosition']['lr'])
    loss_fn = nn.CrossEntropyLoss(ignore_index=config['ClassificationTwoPosition']['ignore_idx']#)
                                  ,label_smoothing=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.2,
                                                           patience=1,
                                                           verbose=True)
    print(f"Vocab: {bert_vocab}")
    inverse_vocab = {v: k for k, v in bert_vocab.items()}
    print(f"Inverse Vocab: {inverse_vocab}")

    #wandb.init(project="two_position_prediction",
    #           config=config)
    
    if config['ClassificationTwoPosition']['perform_inference']:
        save_dict = torch.load(config['ClassificationTwoPosition']['save_path'])
        model.load_state_dict(save_dict['model'])
        optimizer.load_state_dict(save_dict['optimizer'])
        scheduler.load_state_dict(save_dict['scheduler'])
        epoch = save_dict['epoch']
        test(model,
             test_loader,
             device,
             config,
             inverse_vocab,
             data_name=req_save_name)
        exit()
    
    # Train
    best_loss = 1e10
    for epoch in range(config['ClassificationTwoPosition']['epochs']):
        train_accuracy, \
        train_loss, \
        _ = train_forward_two_position_prediction(model,
                                                  train_loader,
                                                  optimizer,
                                                  loss_fn,
                                                  device,
                                                  config)
        test_accuracy, \
        test_loss, \
        test_pred = test_forward_two_position_prediction(model,
                                                         test_loader,
                                                         loss_fn,
                                                         device,
                                                         config)
        scheduler.step(test_loss)
        if test_loss < best_loss:
            best_loss = test_loss
            save_dict = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(),
                         "epoch": epoch,
                         "best_loss": best_loss,
                         "config": config,
                         "best_accuracy": test_accuracy}
            torch.save(save_dict, config['ClassificationTwoPosition']['save_path'])
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_accuracy*100:.2f}%')
        print(f"Epoch: {epoch+1:02} | Test Loss: {test_loss:.3f} | Test Acc: {test_accuracy*100:.2f}%")


if __name__ == '__main__':
    main()