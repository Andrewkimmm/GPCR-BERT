import torch
import torch.nn as nn
import json
import numpy as np
import pandas as pd
from transformers import BertTokenizer, AutoTokenizer
from utils import one_of_k_encoding, \
                  one_of_k_encoding
from tqdm import tqdm


class TwoPositionPredictionDataset(torch.utils.data.Dataset):
    '''
    Dataset for two position prediction
    The input is sequence with two positions masked
    The output is the token in the masked positions
    '''
    def __init__(self,
                 df,
                 config) -> None:
        super().__init__()
        self.df = df
        self.config = config['ClassificationTwoPosition']
        self.tokenizer = BertTokenizer.from_pretrained(config['Bert']['pretrained_model_name'])
        print(f"vocabulary {self.tokenizer.vocab}")

        # # finding an alphabet that is not in the vocabulary
        # capital_letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        # for letter in capital_letters:
        #     if letter not in self.tokenizer.vocab:
        #         print(f"letter {letter} not in vocabulary")

        self.max_len = self.config['max_len']
        self.bert_max_len = self.config['bert_max_len']
        self.my_mask_token = self.config['mask_token']
        self.bert_mask_token = self.config['bert_mask_token']
        #self.req_pre_string = self.config['req_pre_string']
        self.distance_mask_token = self.config['distance_mask_token']
        self.no_mask_token = self.config['no_mask_token']

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        input_full = list(self.df[idx][1])
        input_full_str = ''.join(input_full)
        pad_start = len(input_full_str) + 1 
        # +1 for the [CLS] token

        # findind start and end of req_pre_string
        req_pre = self.df[idx][2]
        req_pre_string = ''.join(req_pre)
        start_idx = input_full_str.find(req_pre_string)
        #start_idx = input_full_str.find(self.req_pre_string)
        
        # print(f"start idx: {start_idx}")
        input_full_str_list = list(input_full_str)
        # replacing the two positions with mask token
        input_full_str_list[start_idx + self.distance_mask_token : \
            start_idx + self.distance_mask_token + self.no_mask_token] = self.my_mask_token * self.no_mask_token
        #input_full_str_list[start_idx+len(self.req_pre_string): 
        #                    start_idx+len(self.req_pre_string)+2] = self.my_mask_token*2
        label_full_str_list = list(input_full_str)
        # replace everything before and after req_pre_string with mask token
        label_full_str_list[:start_idx+self.distance_mask_token] = \
            self.my_mask_token*len(label_full_str_list[:start_idx+self.distance_mask_token])
        label_full_str_list[start_idx+len(req_pre_string) - 1:] = \
            self.my_mask_token*len(label_full_str_list[start_idx+len(req_pre_string)-1:])

        #label_full_str_list[:start_idx+len(self.req_pre_string)] = self.my_mask_token*len(label_full_str_list[:start_idx+len(self.req_pre_string)])
        #label_full_str_list[start_idx+len(self.req_pre_string)+2:] = self.my_mask_token*len(label_full_str_list[start_idx+len(self.req_pre_string)+2:])

        

        #input_full_str = ''.join(input_full_str_list)
        #label_full_str = ''.join(label_full_str_list)
        # print(input_full_str)
        # print(label_full_str)
        # printing number of mask tokens
        # print(f"number of mask tokens: {input_full_str.count(self.my_mask_token)}")
        input_full_str_spaced = ' '.join(input_full_str_list)
        label_full_str_spaced = ' '.join(label_full_str_list)

        input_full_str_spaced = input_full_str_spaced.replace(self.my_mask_token,
                                                              self.bert_mask_token)
        input_full_tokenized = self.tokenizer(input_full_str_spaced,
                                              return_tensors='pt',
                                              padding='max_length',
                                              max_length=self.bert_max_len)
        
        label_full_str_spaced = label_full_str_spaced.replace(self.my_mask_token,
                                                              self.bert_mask_token)
        label_full_tokenized = self.tokenizer(label_full_str_spaced,
                                              return_tensors='pt',
                                              padding='max_length',
                                              max_length=self.bert_max_len)
        # convert label_full_tokenized to 4 to 0
        label_full_tokenized['input_ids'][label_full_tokenized['input_ids'] == 4] = 0
        # convert label_full_tokenized to 2, 3 to 0
        label_full_tokenized['input_ids'][label_full_tokenized['input_ids'] == 2] = 0
        label_full_tokenized['input_ids'][label_full_tokenized['input_ids'] == 3] = 0

        # convert attention of mask to 0 in input_full_tokenized
        input_full_tokenized['attention_mask'][input_full_tokenized['input_ids'] == 4] = 0

        start_idx = start_idx + len(req_pre_string) + 1 # +1 for cls token
        # print(input_full_tokenized['input_ids'][0][start_idx])
        # print(input_full_tokenized['input_ids'][0][start_idx+1])
        # print(label_full_tokenized['input_ids'][0][start_idx])
        # print(label_full_tokenized['input_ids'][0][start_idx+1])
        # print(input_full_tokenized['input_ids'])
        # print(label_full_tokenized['input_ids'])
        if self.config['perform_inference']:
            return input_full_tokenized, \
                   label_full_tokenized, \
                   start_idx, \
                   pad_start
                #   input_full
                #    list(self.df[idx][0]) # pdb name


        return input_full_tokenized, \
               label_full_tokenized, \
               start_idx
               
class PositionPredictionFullDataset(torch.utils.data.Dataset):
    def __init__(self,
                 df,
                 config):
        super().__init__()
        self.df = df # test data = proteins_NPxxY
        self.config = config['ClassificationTwoPosition']
        self.tokenizer = BertTokenizer.from_pretrained(self.config['Bert']['pretrained_model_name'])
        print(f"tokenizer vocab: {self.tokenizer.vocab}")

        self.max_len = self.config['max_len']
        self.bert_max_len = self.config['bert_max_len']
        self.my_mask_token = self.config['mask_token']
        self.bert_mask_token = self.config['bert_mask_token']
        print(self.max_len)
        # self.req_pre_string = self.config['req_pre_string']

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # print(self.df[idx])
        input_str = self.df[idx]['input']
        label_str = self.df[idx]['label']
        # print(f"input_str: {input_str}")
        # print(f"label_str: {label_str}")

        
        start_idx = input_str.find(self.my_mask_token)
        count_of_mask_tokens = input_str.count(self.my_mask_token)
        end_idx = start_idx + count_of_mask_tokens 

        # print(f"start_idx: {start_idx}")
        # print(f"end_idx: {end_idx}")
        # print("#"*20)

        input_str_list = list(input_str)
        input_str_list_spaced = ' '.join(input_str_list)
        label_str_list = list(label_str)
        label_str_list_spaced = ' '.join(label_str_list)

        input_str_list_spaced = input_str_list_spaced.replace(self.my_mask_token,
                                                              self.bert_mask_token)
        input_tokenized = self.tokenizer(input_str_list_spaced,
                                         return_tensors='pt',
                                         padding='max_length',
                                         max_length=self.bert_max_len)
        label_str_list_spaced = label_str_list_spaced.replace(self.my_mask_token,
                                                              self.bert_mask_token)
        label_tokenized = self.tokenizer(label_str_list_spaced,
                                         return_tensors='pt',
                                         padding='max_length',
                                         max_length=self.bert_max_len)
        
        # convert label_tokenized to 4, 3, 2, 1 to 0
        label_tokenized['input_ids'][label_tokenized['input_ids'] == 4] = 0
        label_tokenized['input_ids'][label_tokenized['input_ids'] == 3] = 0
        label_tokenized['input_ids'][label_tokenized['input_ids'] == 2] = 0
        label_tokenized['input_ids'][label_tokenized['input_ids'] == 1] = 0

        # convert attention of mask to 0 in input_tokenized which is 4
        input_tokenized['attention_mask'][input_tokenized['input_ids'] == 4] = 0
        # print(label_tokenized['input_ids'])
        input_vocab = self.tokenizer.convert_ids_to_tokens(input_tokenized['input_ids'][0])
        label_vocab = self.tokenizer.convert_ids_to_tokens(label_tokenized['input_ids'][0])
        # print(input_vocab)
        # print(start_idx, end_idx)
        # for i in range(start_idx+1, end_idx+1):
        #     print(input_vocab[i])
        # print(input_vocab[start_idx+1:end_idx+1])
        # print(label_vocab[start_idx+1:end_idx+1])
        # print("####################"*5)
        # print(label_tokenized['input_ids'][0][start_idx+1:end_idx+1])

        return input_tokenized, \
               label_tokenized, \
               start_idx+1, \
               end_idx+1
               #self.df[idx]['input']


if __name__ == "__main__":
    import yaml
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    # df = pd.read_csv(config['filename'])
    df = np.load(config['ClassificationTwoPosition']['filename'], allow_pickle=True)
    dataset = PositionPredictionFullDataset(df, config)
    # dataset[0]
    # dataset[1]
    # dataset[2]
    # # unit_test_dataset(dataset)
    exit()
    # df = pd.read_csv(config['filename'])
    # filename = config['ClassificationTwoPosition']['filename']
    # df = np.load(filename, allow_pickle=True)
    # dataset = TwoPositionPredictionDataset(df, config)
    # for i in range(10):
    #     dataset[i]
    #     exit()

    # print(config)
    # # filename = config['filename']
    # # df = pd.read_csv(filename)
    # # print(f"colums: {df.columns}")
    
    # # token_ids_filename = "./token_ids.json"
    # # dataset = SingleSequenceHeavyMaskAndLightDataset(df, config)
    # # filename = config['Binds_to_classification']
    # data_np = np.load("./data/binds_to_classification_binary.npy", allow_pickle=True)
    # # y = [data['label'] for data in data_np]
    # from sklearn.model_selection import train_test_split
    
    # train_data, test_data = train_test_split(data_np,
    #                                          test_size=0.25,
    #                                         #  stratify=y,
    #                                          random_state=42)
    
    # # y_train = [data['label'] for data in train_data]
    # # class_weights = torch.zeros(config['Classification']['class_num'])
    # # for y_i in y:
    # #     class_weights[int(y_i)] += 1
    # # class_weights = 1 / class_weights
    # # # normalize the weights
    # # class_weights = class_weights / class_weights.sum()
    # # print(class_weights)
    # # class_weights_for_samples = [class_weights[int(y_i)] for y_i in y_train]
    # # class_weights_for_samples = torch.tensor(class_weights_for_samples)

    # # weighted_random_sampler = torch.utils.data.WeightedRandomSampler(weights=class_weights_for_samples,
    # #                                                                  num_samples=len(class_weights_for_samples),
    # #                                                                  replacement=True)
    

    # dataset = AntibodyClassfication(data_np, config)
    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(dataset, 
    #                         batch_size=4, 
    #                         sampler=None, 
    #                         shuffle=False)

    # # for i in range(len(dataset)):
    # #     tokens, labels = dataset.__getitem__(i)
    #     # break
    # # exit()
    # for i, batch in enumerate(dataloader):
    #     # print(tokens['input_ids'].shape)
    #     # print(labels.shape)
    #     print(f"####################### {i} ############################")
    #     antibody, label = batch
    #     print(antibody['input_ids'].shape)
    #     print(label.shape)
    #     # print(virus['input_ids'].shape)
    #     # print(label.shape)
    #     # pass