import torch
import torch.nn as nn
#from colorama import init
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
import yaml
import json
from tqdm import tqdm
from model import ClassificationTwoPositions, ClassificationTwoPositionsBert
from dataset import TwoPositionPredictionDataset, PositionPredictionFullDataset
#import wandb
from train import train_forward_two_position_prediction, \
                  test_forward_two_position_prediction

#from utils import calculate_num_data_per_class, \
#                  make_colored_string, \
#                 visualize_attention_weights,\
#                  bar_plot  
import json
#from bertviz import head_view
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
        pad_start = batch
        #start_idx,\

        input_tokens['input_ids'] = input_tokens['input_ids'].to(device).squeeze(1)
        input_tokens['attention_mask'] = input_tokens['attention_mask'].to(device).squeeze(1)
        input_tokens['token_type_ids'] = input_tokens['token_type_ids'].to(device).squeeze(1)
        
        label_tokens['input_ids'] = label_tokens['input_ids'].to(device).squeeze(1)
        label_tokens['attention_mask'] = label_tokens['attention_mask'].to(device).squeeze(1)
        label_tokens['token_type_ids'] = label_tokens['token_type_ids'].to(device).squeeze(1)

        with torch.no_grad():
            if config['ClassificationTwoPosition']['use_bert']:
                out, attention = model.forward_test(input_tokens)

                #out, last_hidden_state = model.forward(input_tokens) 
                # attention: (batch_size, num_heads, seq_len, seq_len)
                # input_ids = input_tokens["input_ids"].squeeze(0).tolist()

                # input_tokens_str = test_loader.dataset.tokenizer.convert_ids_to_tokens(input_ids)
            
                # label_tokens_str = test_loader.dataset.tokenizer.convert_ids_to_tokens()
                # print(input_tokens['input_ids'].shape)
                # input_tokens_str = [vocab[int(x)] for x in input_tokens['input_ids'][0].squeeze(0).tolist()]
                # print(input_tokens_str)
                # print(label_tokens_str)
                # exit()3
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
            #print(label_token)
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
            ## print(f"org_str: {org_str[b_no]}")
            
            print(f"Input: {input_str[1:]}")
            req_attention = attention[config['ClassificationTwoPosition']['Bert']['layer_no_to_view']]
            req_attention = req_attention[b_no, 
                                          config['ClassificationTwoPosition']['Bert']['head_no_to_view'],
                                         :,
                                         :].squeeze(0).cpu().numpy()
            # req_attention = req_attention
            print(req_attention.shape)            
            req_pad_start = pad_start[b_no].item()
            #req_attention = req_attention[1:req_pad_start, 1:req_pad_start]
            req_attention = req_attention[1:, 1:]
            print(req_attention.shape)

            print(len(input_str[1:]))
            #print(len(input_str[1:pad_start[b_no.item()]]))

            #print(last_hidden_state.shape)
            #print(last_hidden_state)
            ##print(org_str[b_no])
            #print(len(org_str[b_no]))
            #weight_and_virus.append({"seq": input_str[1:], 
            #                         "state": last_hidden_state})
            weight_and_virus.append({"seq": input_str[1:], 
                                     "attention": req_attention})

            #weight_and_virus.append({"seq": input_str[1:pad_start[b_no].item()], 
            #                         "attention": req_attention})
            ##"seq": org_str[b_no]
            # if curr_idx in req_idxs:
            #     visualize_attention_weights(req_attention,
            #                                 input_str[:req_pad_start],
            #                                 str(curr_idx))

    
    weight_and_virus_np = np.array(weight_and_virus)
    np.save(f"{data_name}_weight_and_sequences_test(head{config['ClassificationTwoPosition']['Bert']['head_no_to_view']}).npy", weight_and_virus_np)
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

    #np.savez("attention.npz", c)

# for inference test on training set
def train(model, 
         train_loader, 
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
    loop = tqdm(train_loader, 
                total=len(train_loader),
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
        pad_start = batch
        #start_idx,\

        input_tokens['input_ids'] = input_tokens['input_ids'].to(device).squeeze(1)
        input_tokens['attention_mask'] = input_tokens['attention_mask'].to(device).squeeze(1)
        input_tokens['token_type_ids'] = input_tokens['token_type_ids'].to(device).squeeze(1)
        
        label_tokens['input_ids'] = label_tokens['input_ids'].to(device).squeeze(1)
        label_tokens['attention_mask'] = label_tokens['attention_mask'].to(device).squeeze(1)
        label_tokens['token_type_ids'] = label_tokens['token_type_ids'].to(device).squeeze(1)

        with torch.no_grad():
            if config['ClassificationTwoPosition']['use_bert']:
                out, attention = model.forward_test(input_tokens)

                #out, last_hidden_state = model.forward(input_tokens) 

        
        for b_no, (input_token, label_token) in enumerate(zip(input_tokens['input_ids'], 
                                                              label_tokens['input_ids'])):
            input_str_list = [vocab[int(x)] for x in input_token.tolist()]
        
            input_str = "".join(input_str_list)
            #print("input_str = ", input_str)
            label_str_list = [vocab[int(x)] for x in label_token.tolist()]
            #print(label_token)
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
            ## print(f"org_str: {org_str[b_no]}")
            
            #print(f"Input: {input_str[1:pad_start[b_no].item()]}")
            print(f"Input: {input_str[1:]}")
            #print(f"Input: {input_str[1:pad_start[b_no].item()]}")

            req_attention = attention[config['ClassificationTwoPosition']['Bert']['layer_no_to_view']]
            # extracting from last layer = 29

            req_attention = req_attention[b_no, 
                                          config['ClassificationTwoPosition']['Bert']['head_no_to_view'],
                                         :,
                                         :].squeeze(0).cpu().numpy()
            # extracting from the first head

            
            req_pad_start = pad_start[b_no].item()
            req_attention = req_attention[1:, 1:]
            
            print(req_attention.shape)
            #print(len(input_str[1:pad_start[b_no].item()]))
            print(len(input_str[1:]))
            #print(last_hidden_state.shape)
            #print(last_hidden_state)
            ##print(org_str[b_no])
            #print(len(org_str[b_no]))
            weight_and_virus.append({"seq": input_str[1:], 
                                     "attention": req_attention})
            #weight_and_virus.append({"seq": input_str[1:pad_start[b_no].item()], 
            #                         "attention": req_attention})
    
    weight_and_virus_np = np.array(weight_and_virus)
    np.save(f"{data_name}_weight_and_sequences_train.npy", weight_and_virus_np)
    print(weight_and_virus_np.shape)
    

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
    train_dataset = PositionPredictionFullDataset(train_data, config)
    test_dataset = PositionPredictionFullDataset(test_data, config)

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
    loss_fn = nn.CrossEntropyLoss(ignore_index=config['ClassificationTwoPosition']['ignore_idx'],
                                  label_smoothing=0.01)
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
        test(model,test_loader,device,config,inverse_vocab, data_name=req_save_name)
        #train(model,train_loader, device,config,inverse_vocab, data_name=req_save_name)
        exit()
    
    # Train
    best_loss = 1e10
    for epoch in range(config['ClassificationTwoPosition']['epochs']):
        #lasthiddenstate
        train_accuracy, \
        train_loss, \
        _ = train_forward_two_position_prediction(model,
                                                  train_loader,
                                                  optimizer,
                                                  loss_fn,
                                                  device,
                                                  config)
        #lasthiddenstate
        test_accuracy, \
        test_loss, \
        _,\
        final_pred,\
        final_gts     = test_forward_two_position_prediction(model,
                                                         test_loader,
                                                         loss_fn,
                                                         device,
                                                         config)
        # for cwxp only
        #if test_accuracy > 0.83:
        #    break

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
        #print(last_hidden_state, last_hidden_state.shape)

    # Alanine scanning
    '''
    alanine_result = []
    for i in range(21):
      test_data2 = np.load(f'drive/MyDrive/PRotBert/alanine(b1)/b1({i}).npy', allow_pickle=True)

      test_dataset2 = PositionPredictionFullDataset(test_data2, config)

      test_loader2 = torch.utils.data.DataLoader(test_dataset2,
                                                batch_size=config['ClassificationTwoPosition']['batch_size'],
                                                shuffle=False)

      test_accuracy, \
          test_loss, \
          _,\
          final_pred,\
          final_gts     = test_forward_two_position_prediction(model,
                                                          test_loader2,
                                                          loss_fn,
                                                          device,
                                                          config)

      
      final_pred = final_pred.cpu().numpy()
      final_gts = final_gts.cpu().numpy()
      #print(final_pred)
      #print(final_gts)

      incorrect_list = []
      for j in range(len(final_pred)):
        if final_pred[j] != final_gts[j]:
          incorrect_list.append(j)
      
      print(f'Beta 1, total length: {len(final_pred)}, Incorrect: {incorrect_list}')
      dict = {"sequence_len": len(final_pred)/2, "final_pred": final_pred, "final_gts": final_gts, "incorrect": incorrect_list}
      alanine_result.append(dict)

    return alanine_result
    '''
    '''
    index_to_amino_acid = {6: 'A', 23: 'C', 14: 'D', 9: 'E', 19: 'F', 7: 'G', 22: 'H', 11: 'I', 
    12: 'K', 5: 'L', 21: 'M', 17: 'N', 16: 'P', 18: 'Q', 13: 'R', 
    10: 'S', 15: 'T', 8: 'V', 24: 'W', 20: 'Y'}
    amino_acids = [index_to_amino_acid[i] for i in range(5,25)]

    y_pred_amino = [index_to_amino_acid[i] for i in final_pred]
    y_gts_amino = [index_to_amino_acid[i] for i in final_gts]

    fig = bar_plot(y_gts_amino, y_pred_amino)
    plt.show()
    '''
'''
accuracy = [1 if y_gts == y_pred else 0 for y_gts, y_pred in zip(y_gts_amino, y_pred_amino)]

# Compute cumulative accuracy
cumulative_accuracy = np.cumsum(accuracy) / np.arange(1, len(accuracy) + 1)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(cumulative_accuracy)
plt.xlabel('Residue Index')
plt.ylabel('Cumulative Accuracy')
plt.title('Per-Residue Accuracy of Protein Sequence Prediction')
plt.grid(True)
plt.show()
'''

# confusion matrix
'''
    conf_matrix = confusion_matrix(y_gts_amino, y_pred_amino, labels=amino_acids)
    #conf_df = pd.DataFrame(conf_matrix, index=amino_acids, columns=amino_acids)
    #bins = 30
    plt.figure(figsize=(10,10))
    #plt.hist2d(final_pred, final_gts, bins=bins, cmap='plasma')
    
    #plt.colorbar(label='count in bin')
    sns.heatmap(conf_matrix, annot = False, xticklabels = amino_acids, yticklabels = amino_acids, fmt = 'd', cmap = 'Blues', square=True)
    plt.gca().invert_yaxis()
    plt.xlabel('Predictions')
    plt.ylabel('Labels')
    plt.title('Task : CWxP')
    plt.show()
'''   
    #mcm = multilabel_confusion_matrix(final_gts, final_pred)

    #for i, label_mcm in enumerate(mcm):
    #    plt.figure(i)
    #    sns.heatmap(label_mcm, annot=True, fmt = 'd', cmap = 'Blues')
    #    plt.xlabel('Predicted')
    #    plt.ylabel('True')
    #    plt.show()

if __name__ == '__main__':
    main()