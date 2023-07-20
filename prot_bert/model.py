import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, \
                         BertModel, \
                         AutoModel, \
                         AutoTokenizer
import json
import yaml
# import pad sequence
from torch.nn.utils.rnn import pad_sequence

# Model for the new taks, of predicting the two positions of the antibody              
class ClassificationTwoPositions(nn.Module):
    def __init__(self,
                 config) -> None:
        super(ClassificationTwoPositions, self).__init__()
        self.config = config['ClassificationTwoPosition']
        self.embedding_layer = EmbeddingLayer(max_len=self.config['Embedding']['bert_max_len'],
                                              vocabulary_size=self.config['Embedding']['vocabulary_size'],
                                              feature_dim=self.config['Embedding']['feature_dim'])
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.config['TransformerEncoder']['d_model'],
                                                        nhead=self.config['TransformerEncoder']['num_heads'],
                                                        dim_feedforward=self.config['TransformerEncoder']['dim_feedforward'],
                                                        batch_first=True,
                                                        dropout=self.config['TransformerEncoder']['dropout'])
        self.encoder = nn.TransformerEncoder(self.encoder_layer,
                                             num_layers=self.config['TransformerEncoder']['num_layers'])
        self.fc = nn.Sequential(nn.Linear(self.config['TransformerEncoder']['d_model'],
                                          self.config['FeedForward']['hidden_dims']),
                                nn.ReLU(),
                                nn.Dropout(self.config['FeedForward']['dropout']),   
                                nn.Linear(self.config['FeedForward']['hidden_dims'],
                                          self.config['FeedForward']['num_classes']))
        
    def forward(self,
                input_tokens,
                input_mask):
        # input_tokens: N*max_len
        # input_mask: N*max_len
        positions_ids = torch.arange(1, input_tokens.shape[1]+1).unsqueeze(0)
        positions_ids = positions_ids.repeat(input_tokens.shape[0], 1).to(input_tokens.device)
        input_features = self.embedding_layer(input_tokens, positions_ids)
        # convert the attention mask to bool, where True means the position is padded
        src_key_padding_mask = input_mask == 0 # N*max_len
        # convert the attention mask to bool, if not bool
        if src_key_padding_mask.dtype != torch.bool:
            src_key_padding_mask = src_key_padding_mask.bool()
        encoded_features = self.encoder(input_features,
                                        src_key_padding_mask=src_key_padding_mask)
        logits = self.fc(encoded_features)
        logits = torch.permute(logits, (0, 2, 1))
        return logits # N*classes*max_len
    
# Model for the new taks, of predicting the two positions of the antibody using BERT as encoder
class ClassificationTwoPositionsBert(nn.Module):
    def __init__(self,
                 config) -> None:
        super(ClassificationTwoPositionsBert, self).__init__()
        self.config = config['ClassificationTwoPosition']
        self.encoder = BertModel.from_pretrained(self.config['Bert']['pretrained_model_name'],
                                                 output_attentions=True if self.config['perform_inference'] else False)

        for key, value in self.encoder.encoder.named_parameters():
            layer_num = int(key.split('.')[1])
            if layer_num < self.config['Bert']['freeze_layers']:
                value.requires_grad = False

        self.fc = nn.Sequential(nn.Linear(self.config['FeedForward']['input_dims'],
                                          self.config['FeedForward']['hidden_dims']),
                                nn.ReLU(),  
                                nn.Dropout(self.config['FeedForward']['dropout']),
                                nn.Linear(self.config['FeedForward']['hidden_dims'],
                                            self.config['FeedForward']['num_classes']))
        
    def forward(self,
                input_tokens):
        # input_tokens: {input_ids, attention_mask, token_type_ids}
        encoded_features = self.encoder(**input_tokens)['last_hidden_state'] # N*max_len*hidden_dims
        logits = self.fc(encoded_features)
        logits = torch.permute(logits, (0, 2, 1))
        return logits # N*classes*max_len
    
    def forward_test(self, 
                     input_tokens):
        # input_tokens: {input_ids, attention_mask, token_type_ids}
        encoded = self.encoder(**input_tokens)
        encoded_features = encoded['last_hidden_state'] # N*max_len*hidden_dims
        attentions = encoded['attentions']
        logits = self.fc(encoded_features)
        logits = torch.permute(logits, (0, 2, 1))
        return logits, attentions # N*classes*max_len, N*num_heads*max_len*max_len
    

if __name__ ==  "__main__":
    config_file = 'config.yaml'
    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

    model = ClassificationTwoPositions(config)
                                        
