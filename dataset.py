import torch
import torch.nn as nn
import json
import numpy as np
import pandas as pd
from transformers import BertTokenizer, AutoTokenizer
from utils import one_of_k_encoding, \
                  one_of_k_encoding, \
                  mol2vec, \
                  atom_features
from rdkit import Chem
from tqdm import tqdm


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 virus,
                 classes,
                 token_ids_filename,
                 config):
        self.virus = virus
        self.classes = classes
        self.config = config
        self.max_len = config['max_len']
        with open(token_ids_filename, 'r') as f:
            self.token_ids = json.load(f)

    def __len__(self):
        return len(self.virus)

    def __getitem__(self, idx):
        concatenated_string = self.virus[idx]
        # print(concatenated_string)
        token_ids = [self.token_ids[id] for id in concatenated_string]
        token_ids = torch.LongTensor(token_ids).reshape(-1)
        pos_ids = [word_no+1 for word_no in range(token_ids.shape[0])]
        attention_mask = torch.ones(token_ids.shape[0], dtype=torch.long)

        # pad with zeros until max_len
        token_ids = torch.cat((token_ids, 
                               torch.zeros(self.max_len - token_ids.shape[0], dtype=torch.long)))
        pos_ids = torch.LongTensor(pos_ids).reshape(-1)
        pos_ids = torch.cat((pos_ids, 
                             torch.zeros(self.max_len - pos_ids.shape[0], dtype=torch.long)))
        attention_mask = torch.cat((attention_mask, 
                                    torch.zeros(self.max_len - attention_mask.shape[0], dtype=torch.long)))
        return token_ids, pos_ids, attention_mask, self.classes[idx]

class CovidDataset(torch.utils.data.Dataset):
    def __init__(self,
                 df,
                 use_heavy=True,
                 token_ids_filename=None,
                 config=None) -> None:
        super(CovidDataset, self).__init__()
        self.df = df
        self.use_heavy = use_heavy
        self.token_ids_filename = token_ids_filename
        self.config = config
        self.virus_max_len = config['virus_max_len']
        self.antibody_max_len = config['vhh_max_len'] if use_heavy else config['vl_max_len']
        self.cdr_max_len = config['cdrh3_max_len'] if use_heavy else config['cdrl3_max_len']
        self.label_pad = config['ignore_label']
        self.num_heads = config['num_heads']
        self.class_count = torch.zeros(config['vocabulary_size'])
        if self.token_ids_filename is not None:
            with open(token_ids_filename, 'r') as f:
                self.token_ids = json.load(f)

        self.inv_token_ids = {int(v): k for k, v in self.token_ids.items()}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        '''
        returns Virus, Antibody, BIOS label
        '''
        virus = self.df.iloc[idx]['Virus']
        antibody = self.df.iloc[idx]['VHorVHH'] if self.use_heavy else self.df.iloc[idx]['VL']
        cdr = self.df.iloc[idx]['CDRH3'] if self.use_heavy else self.df.iloc[idx]['CDRL3']
        label_str = self.df.iloc[idx]['vhh_label_int'] if self.use_heavy else self.df.iloc[idx]['vl_label_int']
        
        virus_token_ids = [self.token_ids[id] for id in virus]
        virus_token_ids = torch.LongTensor(virus_token_ids).reshape(-1)
        virus_pos_ids = [word_no+1 for word_no in range(virus_token_ids.shape[0])]

        if not self.config['is_transformer']:
            virus_attention_mask = torch.ones(virus_token_ids.shape[0], dtype=torch.long)
            virus_attention_mask = torch.cat((virus_attention_mask, 
                                            torch.zeros(self.virus_max_len - virus_attention_mask.shape[0], 
                                                        dtype=torch.long)))
        else:
            # create a mask of self.virus_max_len and True at all positions
            virus_attention_mask = torch.ones(self.virus_max_len, dtype=torch.bool)
            # set false until virus_token_ids.shape[0]
            virus_attention_mask[:virus_token_ids.shape[0]] = False

        # mask with zeros until max_len
        virus_token_ids = torch.cat((virus_token_ids, 
                                     torch.zeros(self.virus_max_len - virus_token_ids.shape[0], 
                                                 dtype=torch.long)))
        virus_pos_ids = torch.LongTensor(virus_pos_ids).reshape(-1)
        virus_pos_ids = torch.cat((virus_pos_ids, 
                                   torch.zeros(self.virus_max_len - virus_pos_ids.shape[0], 
                                               dtype=torch.long)))
        

        antibody_token_ids = [self.token_ids[id] for id in antibody]
        antibody_token_ids = torch.LongTensor(antibody_token_ids).reshape(-1)
        # print(f"antibody_token_ids.shape: {antibody_token_ids.shape}")
        antibody_pos_ids = [word_no+1 for word_no in range(antibody_token_ids.shape[0])]

        if not self.config['is_transformer']:
            antibody_attention_mask = torch.ones(antibody_token_ids.shape[0], dtype=torch.long)
            antibody_attention_mask = torch.cat((antibody_attention_mask,
                                                torch.zeros(self.antibody_max_len - antibody_attention_mask.shape[0],
                                                            dtype=torch.long)))
        else:
            # create a mask of self.antibody_max_len and True at all positions
            antibody_attention_mask = torch.ones(self.antibody_max_len, dtype=torch.bool)
            # set false until antibody_token_ids.shape[0]
            antibody_attention_mask[:antibody_token_ids.shape[0]] = False
            
        antibody_token_ids = torch.cat((antibody_token_ids,
                                        torch.zeros(self.antibody_max_len - antibody_token_ids.shape[0],
                                                    dtype=torch.long)))
        antibody_pos_ids = torch.LongTensor(antibody_pos_ids).reshape(-1)
        antibody_pos_ids = torch.cat((antibody_pos_ids,
                                      torch.zeros(self.antibody_max_len - antibody_pos_ids.shape[0],
                                                  dtype=torch.long)))
        
        
        # cdr_str = self.df.iloc[idx]['CDRH3'] if self.use_heavy else self.df.iloc[idx]['CDRL3']
        # convert string label to list of ints, after removing the first and last characters
        label = [int(c) for c in label_str[1:-1].split(',')]
        label = torch.LongTensor(label).reshape(-1)
        # mask with zeros until max_len with self.label_pad
        label = torch.cat((label,
                           torch.ones(self.antibody_max_len - label.shape[0],
                                      dtype=torch.long) * self.label_pad))

        cross_attention_mask = torch.zeros(self.antibody_max_len, 
                                           self.virus_max_len).bool()
        cross_attention_mask[antibody_token_ids.shape[0]:, :] = True
        cross_attention_mask[:, virus_token_ids.shape[0]:] = True    
        if self.config['is_masked_llm']:
            antibody_token_ids, \
            antibody_token_labels, \
            mask_start_idx, \
            mask_end_idx = self.label_for_masked_llm(label, 
                                                     antibody_token_ids)
            # self.assert_label(mask_start_idx, mask_end_idx, antibody, cdr)
            # self.token_to_string(antibody_token_ids, antibody_token_labels)
            # print (f"virus: {virus_token_ids.shape}, antibody: {antibody_token_ids.shape}")
            return virus_token_ids,  \
                   virus_pos_ids, \
                   virus_attention_mask, \
                   antibody_token_ids, \
                   antibody_pos_ids, \
                   antibody_attention_mask, \
                   cross_attention_mask, \
                   antibody_token_labels, \
                   mask_start_idx, \
                   mask_end_idx
        return virus_token_ids,  \
               virus_pos_ids, \
               virus_attention_mask, \
               antibody_token_ids, \
               antibody_pos_ids, \
               antibody_attention_mask, \
               cross_attention_mask, \
               label

    def label_for_masked_llm(self, 
                             label,
                             antibody_token_ids):
        antibody_token_labels = torch.clone(antibody_token_ids)
        # print(label)
        for id, lab in enumerate(label):
            if lab == 0:
                start_idx = id
            elif lab == 3:
                end_idx = id
                break
        start_idx_tensor = torch.LongTensor([start_idx])
        end_idx_tensor = torch.LongTensor([end_idx])
        antibody_token_ids[start_idx:end_idx+1] = self.token_ids[self.config['mask_token']]
        return antibody_token_ids, antibody_token_labels, start_idx_tensor, end_idx_tensor
    
    def assert_label(self, 
                     start_idx,
                     end_idx, 
                     antibody,
                     cdr):
        cdr_pred = antibody[start_idx:end_idx+1]
        print(f"cdr_pred: {cdr_pred}, org cdr: {cdr}")
        assert cdr == cdr_pred, f"CDR: {cdr}, CDR_"

    def token_to_string(self, masked_token_ids, original_token_ids):
        masked_string = [self.inv_token_ids[int(id.item())] for id in masked_token_ids]
        masked_string = ''.join(masked_string)
        original_string = [self.inv_token_ids[int(id.item())] for id in original_token_ids]
        original_string =''.join(original_string)
        print(f"masked_string: {masked_string}\n, original_string: {original_string}")

class MaskedDataset(torch.utils.data.Dataset):
    '''
    Dataset for masked language model
    Uses custom tokenizer
    Process Antibody and Virus sequences separately
    Returns:
        virus_token_ids: torch.LongTensor
        virus_pos_ids: torch.LongTensor
        virus_attention_mask: torch.BoolTensor
        antibody_token_ids: torch.LongTensor # only non-masked tokens
        antibody_pos_ids: torch.LongTensor
        antibody_attention_mask: torch.BoolTensor
        cross_attention_mask: torch.BoolTensor
        antibody_token_labels: torch.LongTensor # only masked tokens
    '''
    def __init__(self,
                 df,
                 token_ids_filename,
                 config) -> None:
        self.df = df
        with open(token_ids_filename, 'r') as f:
            self.token_ids = json.load(f)
        self.inv_token_ids = {v: k for k, v in self.token_ids.items()}
        self.config = config
        self.use_heavy = config['use_heavy']
        self.virus_max_len = config['virus_max_len']
        self.antibody_max_len = config['vhh_max_len'] if self.use_heavy else config['vl_max_len']
        self.antibody_name = "VHorVHH" if self.use_heavy else "VL"
        self.pad_token = config['pad_token']

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        '''
        returns 
            1. virus_token_ids
            2. virus_pos_ids
            3. virus_attention_mask
            4. antibody_token_ids
            5. antibody_pos_ids
            6. antibody_attention_mask
            7. cross_attention_mask
            8. antibody_token_labels
            9. antibody_masked_labels
        '''
        virus = self.df.iloc[idx]['Virus']
        antibody_full_str = self.df.iloc[idx]["VHorVHH"]
        
        # antibody_full_str = self.df.iloc[idx]['label_vh']
        antibody_masked_str = self.df.iloc[idx]['masked_label_vh']
        
        # find index of self.pad_token in antibody_full_str
        pad_idx = antibody_full_str.find(self.pad_token)
        antibody_attention_mask = torch.zeros(self.antibody_max_len, dtype=torch.bool)
        # print(f"pad_idx: {pad_idx}")
        if pad_idx == -1:
            pad_idx = self.antibody_max_len

        antibody_attention_mask[pad_idx:] = True
        antibody_pos_ids = torch.zeros(self.antibody_max_len, dtype=torch.long)
        antibody_pos_ids[:pad_idx] = torch.arange(1, pad_idx+1)
        
        if self.config['only_train_on_masked']:
            # find start and end of config['mask_token']
            start_mask_idx, end_mask_idx = self.find_start_and_end(antibody_masked_str)
            # construct string of length end_mask_idx - start_mask_idx + 1 with mask_token
            # mask_string = self.config['mask_token'] * (end_mask_idx - start_mask_idx + 1)
            # assert antibody_masked_str[start_mask_idx:end_mask_idx+1] == mask_string

            antibody_full_str_list = list(antibody_full_str)
            for i in range(len(antibody_full_str_list)):
                if i < start_mask_idx or i > end_mask_idx:
                    antibody_full_str_list[i] = self.pad_token
        
            # convert string to list of ints
            antibody_full_str = ''.join(antibody_full_str_list)
            # print(f"antibody_full_str\n: {antibody_full_str}")
            # print(f"antibody_masked_str\n: {antibody_masked_str}")

        # convert string to list of ints
        antibody_token_ids = [self.token_ids[c] for c in antibody_full_str]
        antibody_masked_tokens = [self.token_ids[c] for c in antibody_masked_str]
        antibody_token_ids = torch.LongTensor(antibody_token_ids)
        antibody_masked_tokens = torch.LongTensor(antibody_masked_tokens)

        # print(f"antibody_token_ids\n: {antibody_token_ids}")
        # print(f"antibody_masked_tokens\n: {antibody_masked_tokens}")

    
        virus_token_ids = [self.token_ids[c] for c in virus]
        virus_token_ids = torch.LongTensor(virus_token_ids)
        virus_attention_mask = torch.zeros(self.virus_max_len, dtype=torch.bool)
        virus_attention_mask[len(virus):] = True
        virus_pos_ids = torch.zeros(self.virus_max_len, dtype=torch.long)
        virus_pos_ids[:len(virus)] = torch.arange(1, len(virus)+1)

        cross_attention_mask = torch.zeros((self.antibody_max_len, 
                                            self.virus_max_len), dtype=torch.bool)
        cross_attention_mask[pad_idx:, :] = True
        cross_attention_mask[:, len(virus):] = True
        return virus_token_ids, \
               virus_pos_ids, \
               virus_attention_mask, \
               antibody_token_ids, \
               antibody_pos_ids, \
               antibody_attention_mask, \
               cross_attention_mask, \
               antibody_masked_tokens 

    def find_start_and_end(self,
                           antibody_str):
        start_idx = -1
        end_idx = -1
        for id, lab in enumerate(antibody_str):
            if lab == self.config['mask_token']:
                if start_idx == -1:
                    start_idx = id
                else:
                    end_idx = id
            elif end_idx != -1 and lab != self.config['mask_token']:
                break
        return start_idx, end_idx

class MaskedDatasetBert(torch.utils.data.Dataset):
    '''
    Dataset for masked language modeling
    Uses BertTokenizer
    Process Antibody and Virus sequences separately
    returns
        1. antibody_bert_full_tokens
        2. antibody_bert_masked_tokens
        3. virus_bert_tokens
        4. cross_attention_mask

    '''
    def __init__(self,
                 df,
                 token_ids_filename,
                 config) -> None:
        super().__init__()
        self.df = df
        with open(token_ids_filename, 'r') as f:
            self.token_ids = json.load(f)
        self.config = config
        self.config_bert = config['Bert']
        self.tokenizer = BertTokenizer.from_pretrained(self.config_bert['pretrained_model_name'])
        print(self.tokenizer.vocab)
        self.bert_pad_token = self.config_bert['pad_token']
        self.bert_mask_token = self.config_bert['mask_token']
        self.bert_virus_max_len = self.config_bert['virus_max_len']
        self.bert_antibody_max_len = self.config_bert['vhh_max_len'] if self.config['use_heavy'] else self.config_bert['vl_max_len']
        
        self.my_pad_token = self.config['pad_token']
        self.my_mask_token = self.config['mask_token']
        self.my_virus_max_len = self.config['virus_max_len']
        self.my_antibody_max_len = self.config['vhh_max_len'] if self.config['use_heavy'] else self.config['vl_max_len'] 

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # antibody
        antibody_full_str = self.df.iloc[idx]['VHorVHH'] if self.config['use_heavy'] else self.df.iloc[idx]['VL']
        antibody_masked_str = self.df.iloc[idx]['masked_label_vh'] if self.config['use_heavy'] else self.df.iloc[idx]['masked_label_vl']
        antibody_pad_start_idx = antibody_full_str.find(self.my_pad_token)
        # print("###############################################")
        # # print(antibody_pad_start_idx)
        # print(antibody_full_str)
        # print(antibody_masked_str)
        # print(antibody_pad_start_idx)
        if antibody_pad_start_idx == -1:
            antibody_pad_start_idx = self.my_antibody_max_len

        # antibody_pad_start_idx += 2 # for [CLS] and [SEP]
        # antibody_attention_mask = torch.zeros(self.bert_antibody_max_len,
        #                                       dtype=torch.bool)
        # antibody_attention_mask[antibody_pad_start_idx:] = True

        if self.config['only_train_on_masked']:
            # find start and end of config['mask_token']
            start_mask_idx, end_mask_idx = self.find_start_and_end(antibody_masked_str)

            antibody_full_str_list = list(antibody_full_str)
            for i in range(len(antibody_full_str_list)):
                if i < start_mask_idx or i > end_mask_idx:
                    antibody_full_str_list[i] = self.my_pad_token
            antibody_full_str = ''.join(antibody_full_str_list)

        # trim antibody_full_str and antibody_masked_str by removing the pad tokens
        antibody_full_str = antibody_full_str[:antibody_pad_start_idx]
        antibody_masked_str = antibody_masked_str[:antibody_pad_start_idx]
        # print(antibody_full_str)
        # print(antibody_masked_str)
        ### converting to bert form
        bert_antibody_full_str = antibody_full_str
        bert_antibody_masked_str = antibody_masked_str

        bert_antibody_full_str_spaced = ' '.join(list(bert_antibody_full_str))
        bert_antibody_masked_str_spaced = ' '.join(list(bert_antibody_masked_str))

        bert_antibody_full_str_spaced = bert_antibody_full_str_spaced.replace(self.my_mask_token, 
                                                                              self.bert_mask_token)
        bert_antibody_full_str_spaced = bert_antibody_full_str_spaced.replace(self.my_pad_token, 
                                                                              self.bert_pad_token)
        bert_antibody_masked_str_spaced = bert_antibody_masked_str_spaced.replace(self.my_mask_token, 
                                                                                  self.bert_mask_token)
        bert_antibody_masked_str_spaced = bert_antibody_masked_str_spaced.replace(self.my_pad_token, 
                                                                                  self.bert_pad_token)

        bert_antibody_full_tokens = self.tokenizer(bert_antibody_full_str_spaced,
                                                   return_tensors='pt',
                                                   padding='max_length',
                                                   max_length=self.my_antibody_max_len+2)
        bert_antibody_masked_tokens = self.tokenizer(bert_antibody_masked_str_spaced,
                                                     return_tensors='pt',
                                                     padding='max_length',
                                                     max_length=self.my_antibody_max_len+2)
        # print(bert_antibody_masked_str)
        # print(bert_antibody_masked_tokens['input_ids'])
        # print("after tokenz: ", bert_antibody_full_tokens['input_ids'].shape)
        # print("before: ", len(bert_antibody_full_str))

        # turn [SEP] token into [PAD] token for bert_antibody_full_tokens
        # convert 3 to 0, 2 to 0
        bert_antibody_full_tokens['input_ids'][bert_antibody_full_tokens['input_ids'] == 3] = 0
        bert_antibody_full_tokens['input_ids'][bert_antibody_full_tokens['input_ids'] == 2] = 0
        
        # print(bert_antibody_full_tokens)
        # print(bert_antibody_masked_tokens)
        
        # virus
        virus_full_str = self.df.iloc[idx]['Virus']
        bert_virus_full_str = virus_full_str
        bert_virus_full_str_spaced = ' '.join(list(bert_virus_full_str))
        bert_virus_full_str_spaced = bert_virus_full_str_spaced.replace(self.my_mask_token,
                                                                        self.bert_mask_token)
        bert_virus_full_str_spaced = bert_virus_full_str_spaced.replace(self.my_pad_token,
                                                                        self.bert_pad_token)
        bert_virus_full_tokens = self.tokenizer(bert_virus_full_str_spaced,
                                                return_tensors='pt',
                                                padding='max_length',
                                                max_length=self.my_virus_max_len)

        cross_attention_mask = torch.zeros((self.bert_antibody_max_len, 
                                            self.bert_virus_max_len),
                                            dtype=torch.bool)
        cross_attention_mask[antibody_pad_start_idx+1:, :] = True
        return bert_antibody_full_tokens, \
               bert_antibody_masked_tokens, \
               bert_virus_full_tokens, \
               cross_attention_mask

    def find_start_and_end(self,
                           antibody_str):
        start_idx = -1
        end_idx = -1
        for id, lab in enumerate(antibody_str):
            if lab == self.config['mask_token']:
                if start_idx == -1:
                    start_idx = id
                else:
                    end_idx = id
            elif end_idx != -1 and lab != self.config['mask_token']:
                break
        return start_idx, end_idx
    
class SingleSequenceDataset(torch.utils.data.Dataset):
    '''
    Dataset for single sequence
    Uses BertTokenizer
    Concatenates antibody and virus sequences with [SEP] token
    Returns:
        bert_antibody_full_tokens: tokenized antibody sequence with [SEP] token and masked tokens
        bert_antibody_masked_tokens: tokenized antibody sequence with [SEP] token with only masked tokens (for loss)
    '''
    def __init__(self, 
                 df: pd.DataFrame,
                 config) -> None:
        super().__init__()
        self.config = config  
        self.df = df
        self.config_bert = config['Bert']
        self.tokenizer = BertTokenizer.from_pretrained(self.config_bert['pretrained_model_name'])
        self.bert_vocabulary = self.tokenizer.get_vocab()
        print("bert vocab: ", self.bert_vocabulary)
        self.inverse_bert_vocabulary = {v: k for k, v in self.bert_vocabulary.items()}

        self.my_mask_token = self.config['mask_token']
        self.my_pad_token = self.config['pad_token']
        self.my_sep_token = self.config['sep_token']
    
        self.bert_mask_token = self.config_bert['mask_token']
        self.bert_pad_token = self.config_bert['pad_token']
        self.bert_sep_token = self.config_bert['sep_token']

        self.my_max_total_len = self.config['max_total_len_vh'] if self.config['use_heavy'] else self.config['max_total_len_vl']
        self.bert_max_total_len = self.config_bert['max_total_len_vh'] if self.config['use_heavy'] else self.config_bert['max_total_len_vl']

        self.my_max_antibody_len = self.config['vhh_max_len'] if self.config['use_heavy'] else self.config['vl_max_len']
        self.bert_max_antibody_len = self.config_bert['vhh_max_len'] if self.config['use_heavy'] else self.config_bert['vl_max_len']

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        antibody_full_str = self.df.iloc[idx]['VHorVHH'] if self.config['use_heavy'] else self.df.iloc[idx]['VL']
        antibody_masked_str = self.df.iloc[idx]['masked_label_vh'] if self.config['use_heavy'] else self.df.iloc[idx]['masked_label_vl']
        antibody_pad_start_idx = antibody_full_str.find(self.my_pad_token)
        virus_full_str = self.df.iloc[idx]['Virus']

        if antibody_pad_start_idx == -1:
            antibody_pad_start_idx = self.my_max_antibody_len
        
        start_mask_idx, end_mask_idx = self.find_start_and_end(antibody_masked_str)

        if self.config['only_train_on_masked']:
            antibody_full_str_list = list(antibody_full_str)

            for i in range(len(antibody_full_str_list)):
                if i < start_mask_idx or i > end_mask_idx:
                    antibody_full_str_list[i] = self.my_pad_token
                
            antibody_full_str = ''.join(antibody_full_str_list)

        # trim antibody_full_str and antibody_masked_str
        antibody_full_str = antibody_full_str[:antibody_pad_start_idx]
        antibody_masked_str = antibody_masked_str[:antibody_pad_start_idx]

        # concatenate antibody_masked_str and virus_full_str with my_sep_token
        antibody_virus_masked_str = antibody_masked_str + self.my_sep_token + virus_full_str
        antibody_full_str = antibody_full_str + self.my_pad_token * (len(antibody_virus_masked_str) - len(antibody_full_str))

        # print("==============================================")
        # print(antibody_masked_str)
        # print(antibody_full_str)
        # print(antibody_virus_masked_str)


        bert_antibody_full_str = antibody_full_str
        bert_antibody_virus_masked_str = antibody_virus_masked_str

        bert_antibody_full_str_spaced = ' '.join(list(bert_antibody_full_str))
        bert_antibody_virus_masked_str_spaced = ' '.join(list(bert_antibody_virus_masked_str))

        # replace my_mask_token with bert_mask_token,
        # replace my_pad_token with bert_pad_token,
        # replace my_sep_token with bert_sep_token
        bert_antibody_full_str_spaced = bert_antibody_full_str_spaced.replace(self.my_mask_token, 
                                                                              self.bert_mask_token)
        bert_antibody_full_str_spaced = bert_antibody_full_str_spaced.replace(self.my_pad_token, 
                                                                              self.bert_pad_token)
        bert_antibody_full_str_spaced = bert_antibody_full_str_spaced.replace(self.my_sep_token, 
                                                                              self.bert_sep_token)

        bert_antibody_virus_masked_str_spaced = bert_antibody_virus_masked_str_spaced.replace(self.my_mask_token, 
                                                                                              self.bert_mask_token)
        bert_antibody_virus_masked_str_spaced = bert_antibody_virus_masked_str_spaced.replace(self.my_pad_token, 
                                                                                              self.bert_pad_token)
        bert_antibody_virus_masked_str_spaced = bert_antibody_virus_masked_str_spaced.replace(self.my_sep_token, 
                                                                                              self.bert_sep_token)

        bert_antibody_full_tokens = self.tokenizer(bert_antibody_full_str_spaced,
                                                   return_tensors='pt',
                                                   padding='max_length',
                                                   max_length=self.bert_max_total_len)
        bert_antibody_virus_masked_tokens = self.tokenizer(bert_antibody_virus_masked_str_spaced,
                                                           return_tensors='pt',
                                                           padding='max_length',
                                                           max_length=self.bert_max_total_len)
        # assert bert_antibody_full_tokens['input_ids'].shape == bert_antibody_virus_masked_tokens['input_ids'].shape, f"bert_antibody_full_tokens['input_ids'].shape: {bert_antibody_full_tokens['input_ids'].shape}, bert_antibody_virus_masked_tokens['input_ids'].shape: {bert_antibody_virus_masked_tokens['input_ids'].shape}"
        
        # converting bert_antibody_full_tokens "2", "3" to "0"
        # bert_antibody_full_tokens['input_ids'][bert_antibody_full_tokens['input_ids'] == 2] = 0
        # bert_antibody_full_tokens['input_ids'][bert_antibody_full_tokens['input_ids'] == 3] = 0
        # print(bert_antibody_full_tokens['input_ids'])
        # print(bert_antibody_virus_masked_tokens['input_ids'])


        # replace mask token with pad token in bert_antibody_virus_masked_tokens attention mask
        bert_antibody_virus_masked_tokens['attention_mask'][bert_antibody_virus_masked_tokens['input_ids'] == 4] = 0
        # print(bert_antibody_full_tokens['attention_mask'])
        # print(bert_antibody_virus_masked_tokens['attention_mask'])
        # print(bert_antibody_full_tokens['input_ids'])
        # print(bert_antibody_virus_masked_tokens['input_ids'])
        # print
        return bert_antibody_full_tokens, \
               bert_antibody_virus_masked_tokens
        # antibody_and_virus_str = 

    def find_start_and_end(self,
                           antibody_str):
        start_idx = -1
        end_idx = -1
        for id, lab in enumerate(antibody_str):
            if lab == self.config['mask_token']:
                if start_idx == -1:
                    start_idx = id
                else:
                    end_idx = id
            elif end_idx != -1 and lab != self.config['mask_token']:
                break
        return start_idx, end_idx
    
class SingleSequenceHeavyAndLightDataset(torch.utils.data.Dataset):
    '''
    Similar to SingleSequenceDataset, but uses both heavy and light chains
    '''
    def __init__(self,
                 df, 
                 config) -> None:
        super().__init__()
        self.df = df
        self.config = config
        self.config_bert = self.config['Bert']
        self.tokenizer = BertTokenizer.from_pretrained(self.config_bert['pretrained_model_name'])
        self.bert_vocabulary = self.tokenizer.get_vocab()
        self.inversed_bert_vocabulary = {v: k for k, v in self.bert_vocabulary.items()}
        
        self.my_mask_token = self.config['mask_token']
        self.my_pad_token = self.config['pad_token']
        self.my_sep_token = self.config['sep_token']

        self.bert_mask_token = self.config_bert['mask_token']
        self.bert_pad_token = self.config_bert['pad_token']
        self.bert_sep_token = self.config_bert['sep_token']

        self.bert_max_total_len = self.config_bert['max_total_len_vh_and_vl']

        self.my_max_antibody_len = self.config['vhh_max_len'] if config['use_heavy'] else self.config['vl_max_len']
        self.bert_max_antibody_len = self.config_bert['vhh_max_len'] if config['use_heavy'] else self.config_bert['vl_max_len']

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        antibody_full_str = self.df.iloc[idx]['VHorVHH']
        antibody_masked_str = self.df.iloc[idx]['masked_label_vh']
        antibody_full_second_str = self.df.iloc[idx]['VL']
        virus_full_str = self.df.iloc[idx]['Virus']

        antibody_pad_start_idx = antibody_full_str.find(self.my_pad_token)
        if antibody_pad_start_idx == -1:
            antibody_pad_start_idx = self.my_max_antibody_len

        if self.config['only_train_on_masked']:
            start_mask_idx, end_mask_idx = self.find_start_and_end(antibody_masked_str)
            antibody_full_str_list = list(antibody_full_str)

            for i in range(len(antibody_full_str_list)):
                if i < start_mask_idx or i > end_mask_idx:
                    antibody_full_str_list[i] = self.my_pad_token

            antibody_full_str = "".join(antibody_full_str_list)

        # trim antibody_full_str and antibody_masked_str
        antibody_full_str = antibody_full_str[:antibody_pad_start_idx]
        antibody_masked_str = antibody_masked_str[:antibody_pad_start_idx]

        # find pad token in antibody_full_second_str and trim
        antibody_second_pad_start_idx = antibody_full_second_str.find(self.my_pad_token)
        if antibody_second_pad_start_idx != -1:
            antibody_full_second_str = antibody_full_second_str[:antibody_second_pad_start_idx]

        # making the final req string
        antibody_virus_masked_str = antibody_masked_str + self.my_sep_token + antibody_full_second_str + self.my_sep_token + virus_full_str
        antibody_virus_full_str = antibody_full_str + self.my_pad_token * (len(antibody_virus_masked_str) - len(antibody_full_str)) 

        bert_antibody_virus_masked_str = antibody_virus_masked_str
        bert_antibody_virus_full_str = antibody_virus_full_str

        # making spaced strings
        bert_antibody_virus_masked_str_spaced = ' '.join(list(bert_antibody_virus_masked_str))
        bert_antibody_virus_full_str_spaced = ' '.join(list(bert_antibody_virus_full_str))

        # replace my tokens with bert tokens
        bert_antibody_virus_masked_str_spaced = bert_antibody_virus_masked_str_spaced.replace(self.my_mask_token, self.bert_mask_token)
        bert_antibody_virus_masked_str_spaced = bert_antibody_virus_masked_str_spaced.replace(self.my_pad_token, self.bert_pad_token)
        bert_antibody_virus_masked_str_spaced = bert_antibody_virus_masked_str_spaced.replace(self.my_sep_token, self.bert_sep_token)
        bert_antibody_virus_full_str_spaced = bert_antibody_virus_full_str_spaced.replace(self.my_mask_token, self.bert_mask_token)
        bert_antibody_virus_full_str_spaced = bert_antibody_virus_full_str_spaced.replace(self.my_pad_token, self.bert_pad_token)
        bert_antibody_virus_full_str_spaced = bert_antibody_virus_full_str_spaced.replace(self.my_sep_token, self.bert_sep_token)

        # tokenize
        bert_antibody_virus_full_tokens = self.tokenizer(bert_antibody_virus_full_str_spaced,
                                                         return_tensors='pt',
                                                         padding='max_length',
                                                         max_length=self.bert_max_total_len)
        bert_antibody_virus_masked_tokens = self.tokenizer(bert_antibody_virus_masked_str_spaced,
                                                           return_tensors='pt',
                                                           padding='max_length',
                                                           max_length=self.bert_max_total_len)
        
        # convert bert_antibody_virus_full_tokens "2", "3" to "0"
        bert_antibody_virus_full_tokens['input_ids'][bert_antibody_virus_full_tokens['input_ids'] == 2] = 0
        bert_antibody_virus_full_tokens['input_ids'][bert_antibody_virus_full_tokens['input_ids'] == 3] = 0

        # print(bert_antibody_virus_full_tokens['input_ids'])
        # print(bert_antibody_virus_masked_tokens['input_ids'])

        # print(antibody_virus_masked_str)
        # print(antibody_virus_full_str)
        return bert_antibody_virus_full_tokens, \
               bert_antibody_virus_masked_tokens

    def find_start_and_end(self,
                           antibody_str):
        start_idx = -1
        end_idx = -1
        for id, lab in enumerate(antibody_str):
            if lab == self.config['mask_token']:
                if start_idx == -1:
                    start_idx = id
                else:
                    end_idx = id
            elif end_idx != -1 and lab != self.config['mask_token']:
                break
        return start_idx, end_idx

class SingleSequenceHeavyMaskAndLightDataset(torch.utils.data.Dataset):
    '''
    The input is [MASK]*len(CDRH3) [SEP] LIGHT_CHAIN
    '''
    def __init__(self,
                 df, 
                 config) -> None:
        super().__init__()
        self.df = df
        self.config = config
        self.config_bert = self.config['Bert']
        self.tokenizer = BertTokenizer.from_pretrained(self.config_bert['pretrained_model_name'])
        self.bert_vocabulary = self.tokenizer.get_vocab()

        self.inverse_bert_vocabulary = {v: k for k, v in self.bert_vocabulary.items()}

        self.my_mask_token = self.config['mask_token']
        self.my_pad_token = self.config['pad_token']
        self.my_sep_token = self.config['sep_token']

        self.bert_mask_token = self.config_bert['mask_token']
        self.bert_pad_token = self.config_bert['pad_token']
        self.bert_sep_token = self.config_bert['sep_token']

        self.my_max_total_len = config['max_len_cdrh3_plus_vl'] if config['use_heavy'] else config['max_len_cdrl3_plus_vh']
        if config['use_virus']:
            self.my_max_total_len += config['virus_max_len'] + 1 # for [SEP] token
        
        self.bert_max_total_len = self.my_max_total_len + 2 # for [CLS] and [SEP]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        antibody_first_full_str = self.df.iloc[idx]['VHorVHH']
        antibody_first_masked_str = self.df.iloc[idx]['masked_label_vh']
        antibody_second_full_str = self.df.iloc[idx]['VL']
        virus_full_str = self.df.iloc[idx]['Virus']

        antibody_first_pad_start_idx = antibody_first_full_str.find(self.my_pad_token)
        if antibody_first_pad_start_idx != -1:
            antibody_first_full_str = antibody_first_full_str[:antibody_first_pad_start_idx]
            antibody_first_masked_str = antibody_first_masked_str[:antibody_first_pad_start_idx]
        
        antibody_second_pad_start_idx = antibody_second_full_str.find(self.my_pad_token)
        if antibody_second_pad_start_idx != -1:
            antibody_second_full_str = antibody_second_full_str[:antibody_second_pad_start_idx]

        start_mask_idx, end_mask_idx = self.find_start_and_end(antibody_first_masked_str)
        masked_only_str = antibody_first_masked_str[start_mask_idx:end_mask_idx+1]
        full_str_label = antibody_first_full_str[start_mask_idx: end_mask_idx+1]
        # padding the full_str_label
        full_str_label = full_str_label + self.my_pad_token * (len(antibody_second_full_str) + 1)
        full_str = masked_only_str + self.my_sep_token + antibody_second_full_str
        
        if self.config['use_virus']:
            full_str = full_str + self.my_sep_token + virus_full_str
        
        # replace my tokens with bert tokens
        # making spaced strings
        # print(full_str_label)
        # print(full_str)
        full_str_label_spaced = ' '.join(list(full_str_label))
        full_str_spaced = ' '.join(list(full_str))

        full_str_label_spaced = full_str_label_spaced.replace(self.my_mask_token, 
                                                              self.bert_mask_token)
        full_str_label_spaced = full_str_label_spaced.replace(self.my_pad_token, 
                                                              self.bert_pad_token)
        full_str_label_spaced = full_str_label_spaced.replace(self.my_sep_token, 
                                                              self.bert_sep_token)
        full_str_spaced = full_str_spaced.replace(self.my_mask_token,
                                                  self.bert_mask_token)
        full_str_spaced = full_str_spaced.replace(self.my_pad_token,
                                                  self.bert_pad_token)
        full_str_spaced = full_str_spaced.replace(self.my_sep_token,
                                                  self.bert_sep_token)
        
        bert_input_tokens = self.tokenizer(full_str_spaced,
                                           return_tensors='pt',
                                           padding='max_length',
                                           max_length=self.bert_max_total_len)
        bert_label_tokens = self.tokenizer(full_str_label_spaced,
                                           return_tensors='pt',
                                           padding='max_length',
                                           max_length=self.bert_max_total_len)  
        
        # label tokens convert cls, sep to pad ie "2", "3" to 0
        bert_label_tokens['input_ids'][bert_label_tokens['input_ids'] == 2] = 0
        bert_label_tokens['input_ids'][bert_label_tokens['input_ids'] == 3] = 0

        # set attention mask of mask token to 0, mask token is 4
        bert_input_tokens['attention_mask'][bert_input_tokens['input_ids'] == 4] = 0
        # print(bert_input_tokens['input_ids'])
        # print(bert_label_tokens['input_ids'])
        # print(bert_input_tokens['attention_mask'].shape)
        return bert_label_tokens, bert_input_tokens

    def find_start_and_end(self,
                           antibody_str):
        start_idx = -1
        end_idx = -1
        for id, lab in enumerate(antibody_str):
            if lab == self.config['mask_token']:
                if start_idx == -1:
                    start_idx = id
                else:
                    end_idx = id
            elif end_idx != -1 and lab != self.config['mask_token']:
                break
        return start_idx, end_idx
        

class AutoRegressiveDataset(torch.utils.data.Dataset):
    '''
    This dataset is used for auto regressive training of the model
    Input is Masked Antibody Sequence and virus sequence
    Output is the cdrh3 in an auto regressive manner
    '''
    def __init__(self,
                 df,
                 config) -> None:
        super().__init__()
        self.df = df
        self.config = config
        self.config_bert = self.config['Bert']
        self.tokenizer = BertTokenizer.from_pretrained(self.config_bert['pretrained_model_name'])
        self.my_max_virus_and_vl_len = self.config['max_total_len_vl']
        self.bert_max_virus_and_vl_len = self.config_bert['max_total_len_vl']
        
        self.my_pad_token = self.config['pad_token']
        self.my_mask_token = self.config['mask_token']
        self.my_sep_token = self.config['sep_token']

        self.bert_pad_token = self.config_bert['pad_token']
        self.bert_mask_token = self.config_bert['mask_token']
        self.bert_sep_token = self.config_bert['sep_token']

        self.max_cdr_len = self.config['AutoRegressive']['max_len_cdrh3'] + 1


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,
                    idx):
        antibody_vl_full_str = self.df.iloc[idx]['VL']
        antibody_virus_full_str = self.df.iloc[idx]['Virus']

        vl_pad_start_idx = antibody_vl_full_str.find(self.my_pad_token)
        if vl_pad_start_idx != -1:
            antibody_vl_full_str = antibody_vl_full_str[:vl_pad_start_idx]

        vl_and_virus_full_str = antibody_vl_full_str + self.my_sep_token + antibody_virus_full_str
        # vl_and_virus_full_str = vl_and_virus_full_str + self.my_pad_token * (self.my_max_virus_and_vl_len - len(vl_and_virus_full_str))

        # making spaced strings
        bert_vl_and_virus_full_str_spaced = ' '.join(list(vl_and_virus_full_str))

        # replace my tokens with bert tokens
        bert_vl_and_virus_full_str_spaced = bert_vl_and_virus_full_str_spaced.replace(self.my_mask_token, 
                                                                                      self.bert_mask_token)
        bert_vl_and_virus_full_str_spaced = bert_vl_and_virus_full_str_spaced.replace(self.my_pad_token,
                                                                                      self.bert_pad_token)
        bert_vl_and_virus_full_str_spaced = bert_vl_and_virus_full_str_spaced.replace(self.my_sep_token,
                                                                                      self.bert_sep_token)
        
        # tokenize
        bert_vl_and_virus_full_tokens = self.tokenizer(bert_vl_and_virus_full_str_spaced,
                                                       return_tensors='pt',
                                                       padding='max_length',
                                                       max_length=self.bert_max_virus_and_vl_len)
        
        cdr_full_str = self.df.iloc[idx]['CDRH3']
        # print(len(cdr_full_str))
        cdr_full_str = cdr_full_str + self.my_sep_token + self.my_pad_token * (self.max_cdr_len - len(cdr_full_str) - 1)
        bert_cdr_full_str_spaced = ' '.join(list(cdr_full_str))
        bert_cdr_full_str_spaced = bert_cdr_full_str_spaced.replace(self.my_mask_token,
                                                                    self.bert_mask_token)
        bert_cdr_full_str_spaced = bert_cdr_full_str_spaced.replace(self.my_pad_token,
                                                                    self.bert_pad_token)
        bert_cdr_full_str_spaced = bert_cdr_full_str_spaced.replace(self.my_sep_token,
                                                                    self.bert_sep_token)
        # print(len(bert_cdr_full_str_spaced))
        # tokenize without adding special tokens
        bert_cdr_full_tokens = self.tokenizer(bert_cdr_full_str_spaced,
                                              return_tensors='pt',
                                              padding='max_length',
                                              max_length=self.max_cdr_len,
                                              add_special_tokens=False)
        # print(bert_cdr_full_tokens['input_ids'])
        # print(bert_vl_and_virus_full_tokens['input_ids'])
        return bert_vl_and_virus_full_tokens, \
               bert_cdr_full_tokens

class AntibodyClassfication(torch.utils.data.Dataset):
    '''
    This dataset is used for classification of antibodies
    into types of covid viruses
    Input is Antibody Sequence, and output is the type of virus
    MultiLabel Classification
    '''
    def __init__(self,
                 data,
                 config) -> None:
        super().__init__()
        self.config = config
        self.classification_config = config['Classification']
        self.labels_dict = self.classification_config['labels_dict']
        self.class_num = self.classification_config['class_num']

        if self.classification_config['task'] == 0:
            self.task_config = self.classification_config['Binds_to']
        elif self.classification_config['task'] == 1:
            self.task_config = self.classification_config['Neutralising_classification']
        elif self.classification_config['task'] == 2:
            self.task_config = self.classification_config['Not_Neutralising_classification']

        self.data = data
        self.max_len = self.task_config['max_len']
        self.bert_max_len = self.task_config['bert_max_len']
        self.tokenizer = BertTokenizer.from_pretrained(self.config['Bert']['pretrained_model_name'])
        
        self.my_pad_token = self.config['pad_token']
        self.my_mask_token = self.config['mask_token']
        self.my_sep_token = self.config['sep_token']

        self.bert_pad_token = self.config['Bert']['pad_token']
        self.bert_mask_token = self.config['Bert']['mask_token']
        self.bert_sep_token = self.config['Bert']['sep_token']

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        vh = self.data[idx]['VHorVHH']
        vl = self.data[idx]['VL']
        labels_list = self.data[idx]['label']

        input_str = vh + self.my_sep_token + vl
        input_str_spaced = ' '.join(list(input_str))
        input_str_spaced = input_str_spaced.replace(self.my_mask_token,
                                                    self.bert_mask_token)
        input_str_spaced = input_str_spaced.replace(self.my_pad_token,
                                                    self.bert_pad_token)
        input_str_spaced = input_str_spaced.replace(self.my_sep_token,
                                                    self.bert_sep_token)
        input_tokens = self.tokenizer(input_str_spaced,
                                      return_tensors='pt',
                                      padding='max_length',
                                      max_length=self.bert_max_len)
        labels = torch.zeros(self.class_num)
        for label in labels_list:
            labels[int(label)] = 1

        # print(labels_list)
        # print(labels)
        # print(input_str)
        # print(input_tokens)
        return input_tokens, labels

class AntibodyBindingDataset(torch.utils.data.Dataset):
    '''
    This dataset is used for classification of antobody virus pairs
    into binding and non binding
    or neutralising and non neutralising
    Input is Antibody Sequence and Virus Sequence, and output 
    binary classification
    '''
    def __init__(self,
                 np_data,
                 config) -> None:
        super().__init__()
        self.config = config
        self.np_data = np_data
        self.config_bert = config['Bert']
        self.tokenizer = BertTokenizer.from_pretrained(self.config_bert['pretrained_model_name'])
        self.max_len = config['Classification']['max_len'] if not config['Classification']['Binds_to_classification']['use_cdrh3'] else 228
        self.bert_max_len = self.max_len + 2

        self.my_pad_token = self.config['pad_token']
        self.my_mask_token = self.config['mask_token']
        self.my_sep_token = self.config['sep_token']

        self.bert_pad_token = self.config['Bert']['pad_token']
        self.bert_mask_token = self.config['Bert']['mask_token']
        self.bert_sep_token = self.config['Bert']['sep_token']

    def __len__(self):
        return self.np_data.shape[0]
    
    def __getitem__(self, idx):
        vh = self.np_data[idx]['VHorVHH'] if not self.config['Classification']['Binds_to_classification']['use_cdrh3'] else self.np_data[idx]['CDRH3']
        vl = self.np_data[idx]['VL']
        virus = self.np_data[idx]['Virus']
        label = self.np_data[idx]['label']

        if self.config['Classification']['Binds_to_classification']['use_cdrh3']:
            input_str = vh + self.my_sep_token + virus
        else:
            input_str = vh + self.my_sep_token + vl + self.my_sep_token + virus
        input_str_spaced = ' '.join(list(input_str))
        input_str_spaced = input_str_spaced.replace(self.my_mask_token,
                                                    self.bert_mask_token)
        input_str_spaced = input_str_spaced.replace(self.my_pad_token,
                                                    self.bert_pad_token)
        input_str_spaced = input_str_spaced.replace(self.my_sep_token,
                                                    self.bert_sep_token)
        input_tokens = self.tokenizer(input_str_spaced,
                                      return_tensors='pt',
                                      padding='max_length',
                                      max_length=self.bert_max_len)
        return input_tokens, label
    
class AntibodyBindingWithDecoderDataset(torch.utils.data.Dataset):
    '''
    This dataset is used for classification of antibody virus pairs
    into binding and non binding
    or neutralising and non neutralising
    Uses the decoder to with antibody as query and virus as key
    Input is Antibody Sequence and Virus Sequence, and output
    binary classification
    '''
    def __init__(self,
                 data_np,
                 config) -> None:
        super().__init__()
        self.vhh_max_len = config['vhh_max_len'] if not config['Classification']['Binds_to_classification']['use_cdrh3'] else config['cdrh3_max_len']
        self.virus_max_len = config['virus_max_len']

        self.bert_vhh_max_len = config['vhh_max_len'] + 2
        self.bert_virus_max_len = config['virus_max_len'] + 2

        self.config = config
        self.data_np = data_np

        self.tokenizer = BertTokenizer.from_pretrained(config['Bert']['pretrained_model_name'])

    def __len__(self):
        return self.data_np.shape[0]
    
    def __getitem__(self, idx):
        vhh = self.data_np[idx]['VHorVHH'] if not self.config['Classification']['Binds_to_classification']['use_cdrh3'] else self.data_np[idx]['CDRH3']
        virus = self.data_np[idx]['Virus']
        label = self.data_np[idx]['label']

        vhh_input_str_spaced = ' '.join(list(vhh))
        virus_input_str = virus
        virus_input_str_spaced = ' '.join(list(virus))

        vhh_input_tokens = self.tokenizer(vhh_input_str_spaced,
                                          return_tensors='pt',
                                          padding='max_length',
                                          max_length=self.bert_vhh_max_len)
        virus_input_tokens = self.tokenizer(virus_input_str_spaced,
                                            return_tensors='pt',
                                            padding='max_length',
                                            max_length=self.bert_virus_max_len)
        return vhh_input_tokens, virus_input_tokens, label                                                            

class AntibodyBindingWithGraph(torch.utils.data.Dataset):
    '''
    This dataset is used for classification of antibody virus pairs
    into binding and non binding
    or neutralising and non neutralising
    Uses rdkit to generate graph features for antibody and virus
    Input is Antibody Sequence and Virus Sequence, and output
    binary classification
    '''
    def __init__(self,
                 data_np,
                 config) -> None:
        super().__init__()
        self.data_np = data_np
        if config['Classification']['task'] == 3:
            self.task = 'Binds_to_classification'
        else:
            self.task = 'Neutralising'
        self.max_len = config['Classification'][self.task]['max_atom_len'] 
        self.config = config
        self.graph_feat_size = 37
        self.max_cdrh3_atom_len = config['Classification'][self.task]['max_cdrh3_atom_len']
        self.max_virus_atom_len = config['Classification'][self.task]['max_virus_atom_len']

        # for i in tqdm(range(len(data_np))):
            
    def __len__(self):
        return self.data_np.shape[0]
    
    def __getitem__(self, idx):
        virus = self.data_np[idx]['Virus']
        label = self.data_np[idx]['label']
        cdrh3 = self.data_np[idx]['CDRH3']

        virus_mol = Chem.MolFromFASTA(virus)
        virus_vec = mol2vec(virus_mol)
        adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(virus_mol)
        virus_features = adjacency_matrix @ virus_vec

        cdrh3_mol = Chem.MolFromFASTA(cdrh3)
        cdrh3_vec = mol2vec(cdrh3_mol)
        adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(cdrh3_mol)
        cdrh3_features = adjacency_matrix @ cdrh3_vec

        cdrh3_features = torch.tensor(cdrh3_features, dtype=torch.float32) # (cdrh3_len * 37)
        virus_features = torch.tensor(virus_features, dtype=torch.float32)

        virus_mask = torch.ones((self.max_virus_atom_len)).to(torch.bool)
        cdrh3_mask = torch.ones((self.max_cdrh3_atom_len)).to(torch.bool)
        virus_mask[:virus_features.shape[0]] = False
        cdrh3_mask[:cdrh3_features.shape[0]] = False

        virus_pad = torch.zeros((self.max_virus_atom_len - virus_features.shape[0], self.graph_feat_size), dtype=torch.float32)
        cdrh3_pad = torch.zeros((self.max_cdrh3_atom_len - cdrh3_features.shape[0], self.graph_feat_size), dtype=torch.float32)

        virus_features = torch.cat([virus_features, virus_pad], dim=0)
        cdrh3_features = torch.cat([cdrh3_features, cdrh3_pad], dim=0)

        return cdrh3_features, \
               virus_features, \
               cdrh3_mask, \
               virus_mask, \
               label                                               

class AutoRegressiveEncoderDataset(torch.utils.data.Dataset):
    '''
    This dataset is used for training the auto regressive encoder
    Input is Antibody Sequence which is masked, and output is
    the masked cdrh3 sequence
    This is done in an auto regressive manner
    '''
    def __init__(self,
                 df,
                 config) -> None:
        super().__init__()
        self.df = df
        self.config = config
        self.max_len = config['max_total_len_vh']
        self.vhh_max_len = config['vhh_max_len'] 
        self.tokenizer = BertTokenizer.from_pretrained(config['Bert']['pretrained_model_name'])
        self.bert_max_len = config['max_total_len_vh'] + 2

        self.my_pad_token = self.config['pad_token']
        self.my_mask_token = self.config['mask_token']
        self.my_sep_token = self.config['sep_token']

        self.bert_pad_token = self.config['Bert']['pad_token']
        self.bert_mask_token = self.config['Bert']['mask_token']
        self.bert_sep_token = self.config['Bert']['sep_token']

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        virus_full_str = self.df.iloc[idx]['Virus']
        vhh_full_str = self.df.iloc[idx]['VHorVHH']
        vhh_masked_str = self.df.iloc[idx]['masked_label_vh']
        start_mask_idx, end_mask_idx = self.find_start_and_end(vhh_masked_str)
        mask_len = end_mask_idx - start_mask_idx + 1
        vhh_pad_start_idx = vhh_full_str.find(self.my_pad_token)

        if vhh_pad_start_idx != -1:
            vhh_full_str = vhh_full_str[:vhh_pad_start_idx]
            vhh_masked_str = vhh_masked_str[:vhh_pad_start_idx]
        
        vhh_and_virus_str = vhh_masked_str + self.my_sep_token + virus_full_str
        vhh_full_str = vhh_full_str + self.my_pad_token * (len(vhh_and_virus_str) - len(vhh_full_str))
        # print(vhh_and_virus_str)
        # print(vhh_full_str)
        vhh_and_virus_str_spaced = ' '.join(vhh_and_virus_str)
        vhh_full_str_spaced = ' '.join(vhh_full_str)

        # replace my tokens with bert tokens
        vhh_and_virus_str_spaced = vhh_and_virus_str_spaced.replace(self.my_pad_token, 
                                                                    self.bert_pad_token)
        vhh_and_virus_str_spaced = vhh_and_virus_str_spaced.replace(self.my_mask_token,
                                                                    self.bert_mask_token)   
        vhh_and_virus_str_spaced = vhh_and_virus_str_spaced.replace(self.my_sep_token,
                                                                    self.bert_sep_token)
        
        vhh_full_str_spaced = vhh_full_str_spaced.replace(self.my_pad_token,
                                                          self.bert_pad_token)
        vhh_full_str_spaced = vhh_full_str_spaced.replace(self.my_mask_token,
                                                          self.bert_mask_token)
        vhh_full_str_spaced = vhh_full_str_spaced.replace(self.my_sep_token,
                                                          self.bert_sep_token)
        
        # tokenize
        vhh_and_virus_tokenized = self.tokenizer(vhh_and_virus_str_spaced,
                                                 return_tensors='pt',
                                                 padding='max_length',
                                                 max_length=self.bert_max_len)
        vhh_full_tokenized = self.tokenizer(vhh_full_str_spaced,
                                            return_tensors='pt',
                                            padding='max_length',
                                            max_length=self.bert_max_len)
        
        #convert vhh_full 2, 3 to 0
        vhh_full_tokenized['input_ids'][vhh_full_tokenized['input_ids'] == 2] = 0
        # vhh_full_tokenized['input_ids'][vhh_full_tokenized['input_ids'] == 3] = 0
        
        # until start of mask make vhh_full 0
        vhh_full_tokenized['input_ids'][:, :start_mask_idx+1] = 0
        # after end of mask make vhh_full 0
        vhh_full_tokenized['input_ids'][:, end_mask_idx+1:] = 0
        
        # print(vhh_full_tokenized['input_ids'])
        # print(vhh_and_virus_tokenized['input_ids'])
        # print(start_mask_idx)
        # print(mask_len)
        # print(end_mask_idx)
        # print("@@@@@@@@@@@#####################!!!!!!!!!#########")

        return {'vhh_and_virus_tokenized': vhh_and_virus_tokenized,
                'vhh_full_tokenized': vhh_full_tokenized,
                'mask_start_idx': start_mask_idx,
                'mask_end_idx': end_mask_idx,
                'mask_len': mask_len}

    def find_start_and_end(self,
                           antibody_str):
        start_idx = -1
        end_idx = -1
        for id, lab in enumerate(antibody_str):
            if lab == self.config['mask_token']:
                if start_idx == -1:
                    start_idx = id
                else:
                    end_idx = id
            elif end_idx != -1 and lab != self.config['mask_token']:
                break
        return start_idx, end_idx

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

class MaskedDatasetAbLangHeavy(torch.utils.data.Dataset):
    def __init__(self, df, config):
        self.config = config['AbLang']
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['pretrained_model_name'])
        print(f"tokenizer vocab: {self.tokenizer.vocab}")
        

        self.max_len = self.config['max_len']
        self.ab_max_len = self.config['ablang_max_len']

        self.mask_token = self.config['mask_token']
        self.pad_token = self.config['pad_token']
        self.sep_token = self.config['sep_token']

        self.ablang_mask_token = self.config['ablang_mask_token']
        self.ablang_pad_token = self.config['ablang_pad_token']
        self.ablang_sep_token = self.config['ablang_sep_token']

        self.ablang_mask_token_id = self.tokenizer.convert_tokens_to_ids(self.ablang_mask_token)
        self.ablang_pad_token_id = self.tokenizer.convert_tokens_to_ids(self.ablang_pad_token)
        self.ablang_sep_token_id = self.tokenizer.convert_tokens_to_ids(self.ablang_sep_token)
        self.ablang_cls_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        antibody_full_str = self.df.iloc[idx]['VHorVHH']
        antibody_masked_str = self.df.iloc[idx]['masked_label_vh']

        pad_start = antibody_full_str.find(self.pad_token)        
        antibody_full_str = antibody_full_str[:pad_start] if pad_start != -1 else antibody_full_str
        antibody_masked_str = antibody_masked_str[:pad_start] if pad_start != -1 else antibody_masked_str
        
        start_idx, end_idx = self.find_start_and_end(antibody_masked_str)
        antibody_full_str_list = list(antibody_full_str)
        antibody_full_str_list[:start_idx] = self.config['mask_token']*len(antibody_full_str_list[:start_idx])
        antibody_full_str_list[end_idx+1:] = self.config['mask_token']*len(antibody_full_str_list[end_idx+1:])
        antibody_full_str = ''.join(antibody_full_str_list)

        # print(f"antibody_full_str:   {antibody_full_str}")
        # print(f"antibody_masked_str: {antibody_masked_str}")

        antibody_full_str_spaced = ' '.join(list(antibody_full_str))
        antibody_masked_str_spaced = ' '.join(list(antibody_masked_str))
        
        antibody_full_str_spaced = antibody_full_str_spaced.replace(self.mask_token,
                                                                    self.ablang_mask_token)
        antibody_masked_str_spaced = antibody_masked_str_spaced.replace(self.mask_token,
                                                                        self.ablang_mask_token)
        antibody_full_tokenized = self.tokenizer(antibody_full_str_spaced,
                                                 return_tensors='pt',
                                                 padding='max_length',
                                                 max_length=self.ab_max_len)
        antibody_masked_tokenized = self.tokenizer(antibody_masked_str_spaced,
                                                   return_tensors='pt',
                                                   padding='max_length',
                                                   max_length=self.ab_max_len)
        
        # convert mask, sep, cls, pad tokens to 0 so it doesn't affect loss
        antibody_full_tokenized['input_ids'][antibody_full_tokenized['input_ids'] == self.ablang_mask_token_id] = 0
        antibody_full_tokenized['input_ids'][antibody_full_tokenized['input_ids'] == self.ablang_sep_token_id] = 0
        antibody_full_tokenized['input_ids'][antibody_full_tokenized['input_ids'] == self.ablang_cls_token_id] = 0
        antibody_full_tokenized['input_ids'][antibody_full_tokenized['input_ids'] == self.ablang_pad_token_id] = 0

        # convert attention mask of antibody_masked_tokenized to 0, where input_ids is self.ablang_mask_token_id
        antibody_masked_tokenized['attention_mask'][antibody_masked_tokenized['input_ids'] == self.ablang_mask_token_id] = 0

        antibody_masked_tokenized_input_ids_string = ' '.join([str(x.item()) for x in antibody_masked_tokenized['input_ids'][0]])
        antibody_full_tokenized_input_ids_string = ' '.join([str(x.item()) for x in antibody_full_tokenized['input_ids'][0]])

        # print(f"antibody_masked_tokenized_input_ids_string: {antibody_masked_tokenized_input_ids_string}")
        # print(f"antibody_full_tokenized_input_ids_string:   {antibody_full_tokenized_input_ids_string}")

        # print(f"antibody_full_str:   {antibody_full_str}")
        # print(f"antibody_masked_str: {antibody_masked_str}")
        return antibody_masked_tokenized, antibody_full_tokenized

    def find_start_and_end(self,
                           antibody_str):
        start_idx = -1
        end_idx = -1
        for id, lab in enumerate(antibody_str):
            if lab == self.config['mask_token']:
                if start_idx == -1:
                    start_idx = id
                else:
                    end_idx = id
            elif end_idx != -1 and lab != self.config['mask_token']:
                break
        return start_idx, end_idx

def unit_test_dataset(dataset):
    for i in tqdm(range(len(dataset)),
                  total=len(dataset)):
        # print(f"i: {i}")
        dataset[i]
    print("unit test passed for dataset")


if __name__ == "__main__":
    import yaml
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    # df = pd.read_csv(config['filename'])
    df = np.load(config['ClassificationTwoPosition']['filename'], allow_pickle=True)
    dataset = PositionPredictionFullDataset(df, config)
    unit_test_dataset(dataset)
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