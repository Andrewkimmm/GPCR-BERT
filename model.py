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

# add classes to import to __init__.py


# Models to predict the CDR3 sequence from the antibody and virus sequences
class EmbeddingLayer(nn.Module):
    '''
    Embedding layer for the model
    Does the following:
    1. Embeds the input sequence
    2. Adds positional embeddings
    3. Applies layer normalization
    args:
        max_len: maximum length of the sequence
        vocabulary_size: size of the vocabulary
        feature_dim: dimension of the embedding
    '''
    def __init__(self,
                 max_len,
                 vocabulary_size,
                 feature_dim) -> None:
        super(EmbeddingLayer, self).__init__()
        print(f"max_len: {max_len}")
        print(f"vocabulary_size: {vocabulary_size}")
        self.acid_embedding = nn.Embedding(vocabulary_size, feature_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len+1, feature_dim, padding_idx=0)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, 
                input_ids, 
                position_ids):
        acid_embedding = self.acid_embedding(input_ids)
        pos_embedding = self.pos_embedding(position_ids)
        embeddings = acid_embedding + pos_embedding
        embeddings = self.layer_norm(embeddings)
        return embeddings

class EncoderLayer(nn.Module):
    '''
    Encoder layer for the model using PyTorch's TransformerEncoder
    args:
        num_layers: number of layers in the encoder
        d_model: dimension of the model
        n_heads: number of heads in the multi-head attention
        dim_feedforward: dimension of the feedforward layer inside the transformer layer
    '''
    def __init__(self,
                 num_layers,
                 d_model,
                 n_heads,
                 dim_feedforward) -> None:
        super(EncoderLayer, self).__init__()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                                    nhead=n_heads,
                                                                    dim_feedforward=dim_feedforward,
                                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer,
                                                         num_layers=num_layers,
                                                         enable_nested_tensor=False)
        
    def forward(self,
                src,
                mask=None,
                src_key_padding_mask=None):
        out = self.transformer_encoder(src,
                                       src_key_padding_mask=mask)
        return out

class DecoderLayer(nn.Module):
    '''
    Decoder layer for the model using PyTorch's TransformerDecoder
    args:
        num_layers: number of layers in the decoder 
        d_model: dimension of the model
        n_heads: number of heads in the multi-head attention
        dim_feedforward: dimension of the feedforward layer inside the transformer layer
    '''

    def __init__(self,
                 num_layers,
                 d_model,
                 n_heads,
                 dim_feedforward) -> None:
        super(DecoderLayer, self).__init__()
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                                    nhead=n_heads,
                                                                    dim_feedforward=dim_feedforward,
                                                                    batch_first=True)
        
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer,
                                                         num_layers=num_layers)

class Model(nn.Module):
    '''
    Model class for the model that uses pytorch's transformer encoder and decoder layers
    encodes the virus and antibody sequences and then decodes the antibody sequence
    with the virus sequence as the context (cross-attention)
    args:
        config: configuration file for the model
    '''
    def __init__(self,
                 config) -> None:
        super(Model, self).__init__()
        embedding_config = config['Embedding']
        transformer_encoder_config = config['TransformerEncoder']
        transformer_decoder_config = config['TransformerDecoder']
        feedforward_config = config['FeedForward']
        self.decoder_num_heads = config['TransformerDecoder']['num_heads']

        self.embedding = EmbeddingLayer(embedding_config['max_pos_len'],
                                        embedding_config['vocabulary_size'],
                                        embedding_config['feature_dim'])
        self.encoder = EncoderLayer(transformer_encoder_config['num_layers'],
                                    transformer_encoder_config['d_model'],
                                    transformer_encoder_config['num_heads'],
                                    transformer_encoder_config['dim_feedforward'])
        
        self.decoder = nn.MultiheadAttention(transformer_decoder_config['d_model'],
                                             transformer_decoder_config['num_heads'],
                                             batch_first=True)

        self.fc = nn.Sequential(nn.Linear(transformer_decoder_config['d_model'], 
                                          feedforward_config['hidden_dims']),
                                nn.ReLU(),
                                nn.Linear(feedforward_config['hidden_dims'], 
                                          feedforward_config['num_classes']))

    def forward(self,
                virus_input_ids,
                virus_position_ids,
                virus_attention_mask,
                antibody_input_ids,
                antibody_position_ids,
                antibody_attention_mask,
                cross_attention_mask=None):
        '''
        forward pass of the model
        args:
            virus_input_ids: input ids for the virus sequence from bert tokenizer
            virus_position_ids: position ids for the virus sequence from bert tokenizer
            virus_attention_mask: attention mask for the virus sequence from bert tokenizer
            antibody_input_ids: input ids for the antibody sequence from bert tokenizer
            antibody_position_ids: position ids for the antibody sequence from bert tokenizer
            antibody_attention_mask: attention mask for the antibody sequence from bert tokenizer
            cross_attention_mask: cross attention mask for the antibody sequence from bert tokenizer
        NB: some of the code is commented out, where the decoder is not used, only the encoder is used
        '''
        virus_embeddings = self.embedding(virus_input_ids,
                                          position_ids=virus_position_ids)
        antibody_embeddings = self.embedding(antibody_input_ids,
                                             position_ids=antibody_position_ids)
        # print(f"virus_embeddings:\n {virus_embeddings}")
        # print(f"antibody_embeddings:\n {antibody_embeddings}")
        # # exit()
        # virus_features = self.encoder(virus_embeddings,
        #                               mask=virus_attention_mask)
        antibody_features = self.encoder(antibody_embeddings,
                                         mask=antibody_attention_mask)
        
        # cross_attention_mask = cross_attention_mask.repeat(self.decoder_num_heads, 1, 1)
        # cross_features = self.decoder(antibody_features,
        #                               virus_features,
        #                               virus_features,
        #                               attn_mask=None,
        #                               need_weights=False)[0]
        logits = self.fc(antibody_features)
        logits = torch.permute(logits, (0, 2, 1))
        return logits

class ModelBert(nn.Module):
    '''
    Model class for the model that uses Bert encoder and pytorch multihead attention layers
    encodes the virus and antibody sequences and then decodes the antibody sequence
    with the virus sequence as the context (cross-attention)
    args:
        config: configuration file for the model
    NB: some of the code is commented out, where the multihead is not used, only the encoder is used
    '''
    def __init__(self,
                 config) -> None:
        super(ModelBert, self).__init__()
        self.config = config
        self.virus_encoder = BertModel.from_pretrained(config['Bert']['pretrained_model_name'])
        for key, value in self.virus_encoder.named_parameters():
            # print(key)   
            # if 'embedding' in key or 'pooler' in key:
            #     value.requires_grad = False
            #     continue
            value.requires_grad = False
        
        self.antibody_encoder = BertModel.from_pretrained(config['Bert']['pretrained_model_name'])
        for key, value in self.antibody_encoder.encoder.named_parameters():
            # if 'embedding' in key or 'pooler' in key:
            #     value.requires_grad = False
            #     continue
            layer_no = int(key.split('.')[1])
            if layer_no < 10:
                value.requires_grad = False

        self.decoder = nn.MultiheadAttention(config['TransformerDecoder']['d_model'],
                                             config['TransformerDecoder']['num_heads'],
                                             batch_first=True)
        self.fc = nn.Sequential(nn.Linear(config['TransformerDecoder']['d_model'],
                                          config['FeedForward']['hidden_dims']),
                                nn.ReLU(),
                                nn.Linear(config['FeedForward']['hidden_dims'],
                                          config['Bert']['num_classes']))
        
    def forward(self,
                antibody_full_tokens,
                anitbody_masked_tokens,
                virus_full_tokens=None,
                cross_attention_mask=None):
        # print(anitbody_masked_tokens['input_ids'].shape)
        antibody_features = self.antibody_encoder(**anitbody_masked_tokens)['last_hidden_state']
        # virus_features = self.virus_encoder(**virus_full_tokens)['last_hidden_state']
        # cross_attention_mask = cross_attention_mask.repeat(self.config['TransformerDecoder']['num_heads'], 
        #                                                    1, 1)
        # # print(antibody_features.shape, virus_features.shape)
        # cross_features = self.decoder(antibody_features,
                                    #   virus_features,
                                    #   virus_features,
                                    #   attn_mask=None)[0]
        logits = self.fc(antibody_features)
        logits = torch.permute(logits, (0, 2, 1))
        return logits

class ModelBertAntibodyAndVirus(nn.Module):
    '''
    Model class for the model that uses only Bert encoder layers    
    concatenates the virus and antibody sequences and then encodes them
    args:
        config: configuration file for the model
    '''
    def __init__(self,
                 config) -> None:
        super(ModelBertAntibodyAndVirus, self).__init__()
        self.config = config
        self.encoder = BertModel.from_pretrained(config['Bert']['pretrained_model_name'])

        for key, value in self.encoder.embeddings.named_parameters():
            value.requires_grad = False

        for key, value in self.encoder.encoder.named_parameters():
            # print(key)
            layer_no = int(key.split('.')[1])
            if layer_no < config['Bert']['start_layer'] or layer_no > config['Bert']['end_layer']:
                value.requires_grad = False

        self.fc = nn.Sequential(nn.Linear(config['FeedForward']['input_dims'],
                                          config['FeedForward']['hidden_dims']),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(config['FeedForward']['hidden_dims'],
                                          config['Bert']['num_classes']))

    def forward(self,
                antibody_and_virus_mask_tokens):
        features = self.encoder(**antibody_and_virus_mask_tokens)['last_hidden_state']
        logits = self.fc(features)
        logits = torch.permute(logits, (0, 2, 1))
        return logits
                                          
class CovidCDRModel(nn.Module):
    '''
    Model class for the model that uses Bert encoder and pytorch multihead attention layers
    encodes the virus and antibody sequences separately 
    cross attention is used where antibody sequence is used as the 
    query and virus sequence is used as the key and value
    args:
        config: configuration file for the model
    NB: Only uses MultiheadAttention layer, no decoder layer
    '''
    def __init__(self,
                 config) -> None:
        super(CovidCDRModel, self).__init__()
        self.config = config
        self.embedding = EmbeddingLayer(config['virus_max_len'],
                                        config['vocabulary_size'],
                                        config['feature_dim'])
        self.virus_encoder = BertModel.from_pretrained('Rostlab/prot_bert')
        self.virus_encoder.embeddings = self.embedding

        self.antibody_encoder = BertModel.from_pretrained('Rostlab/prot_bert')
        self.antibody_encoder.embeddings = self.embedding

        for key, param in self.virus_encoder.encoder.named_parameters():
            layer_no = int(key.split('.')[1])
            if layer_no < 25 and layer_no > 5:
                param.requires_grad = False
            # param.requires_grad = False
            print(key, param.shape, param.requires_grad)
        
        for key, param in self.antibody_encoder.encoder.named_parameters():
            layer_no = int(key.split('.')[1])
            if layer_no < 25 and layer_no > 5:
                param.requires_grad = False
            # param.requires_grad = False
            print(key, param.shape, param.requires_grad)
            # param.requires_grad = False
            # print(key, param.shape, param.requires_grad)
        if config['is_masked_llm']:
            self.fc = nn.Sequential(nn.Linear(1024, config['hidden_dims']),
                                    nn.ReLU(),
                                    nn.Linear(config['hidden_dims'], config['vocabulary_size']))
        else:
            self.fc = nn.Sequential(nn.Linear(1024, config['hidden_dims']),
                                    nn.ReLU(),
                                    nn.Linear(config['hidden_dims'], config['num_classes']))
        self.cross_attention = nn.MultiheadAttention(embed_dim=1024,
                                                     num_heads=config['num_heads'],
                                                     batch_first=True)
        
    def forward(self,
                virus_input_ids,
                virus_position_ids,
                virus_attention_mask,
                antibody_input_ids,
                antibody_position_ids,
                antibody_attention_mask,
                cross_attention_mask=None):
        virus_out = self.virus_encoder(input_ids=virus_input_ids,
                                       attention_mask=virus_attention_mask,
                                       position_ids=virus_position_ids)
        virus_out = virus_out['last_hidden_state']

        antibody_out = self.antibody_encoder(input_ids=antibody_input_ids,
                                             attention_mask=antibody_attention_mask,
                                             position_ids=antibody_position_ids)
        antibody_out = antibody_out['last_hidden_state']
        # print(f"virus_out: {virus_out.shape}")
        # print(f"antibody_out: {antibody_out.shape}")
        # repeat cross attention mask for each head
        cross_attention_mask = cross_attention_mask.repeat(self.config['num_heads'], 1, 1) if cross_attention_mask is not None else None
        out = self.cross_attention(antibody_out, 
                                   virus_out,
                                   virus_out,
                                   attn_mask=cross_attention_mask)
        out = out[0]
        # print(f"out: {out.shape}")
        out = self.fc(out)
        # print(f"out: {out.shape}")
        # exchange dim for cross entropy loss
        out = torch.permute(out, (0, 2, 1)) 
        return out

class BertWithDecoder(nn.Module):
    '''
    Model class for the model that uses Bert encoder and pytorch transformer decoder layers
    encodes the virus and antibody sequences separately
    Uses the transformer decoder layer to decode the antibody sequence
    args:
        config: configuration file for the model
    '''
    def __init__(self, 
                 config) -> None:
        super(BertWithDecoder, self).__init__()
        self.config = config
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=config['TransformerDecoder']['d_model'],
                                                                    nhead=config['TransformerDecoder']['num_heads'],
                                                                    dim_feedforward=config['TransformerDecoder']['dim_feedforward'],
                                                                    batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer,
                                                         num_layers=config['TransformerDecoder']['num_layers'])
        self.virus_encoder = BertModel.from_pretrained(config['Bert']['pretrained_model_name'])
        self.antibody_encoder = BertModel.from_pretrained(config['Bert']['pretrained_model_name'])
        
        for key, value in self.virus_encoder.embeddings.named_parameters():
            value.requires_grad = False

        for key, value in self.virus_encoder.encoder.named_parameters():
            layer_no = int(key.split('.')[1])
            if layer_no < config['Bert']['start_layer'] or layer_no > config['Bert']['end_layer']:
                value.requires_grad = False

        for key, value in self.antibody_encoder.embeddings.named_parameters():
            value.requires_grad = False

        for key, value in self.antibody_encoder.encoder.named_parameters():
            layer_no = int(key.split('.')[1])
            if layer_no < config['Bert']['start_layer'] or layer_no > config['Bert']['end_layer']:
                value.requires_grad = False
        
        
        self.fc = nn.Sequential(nn.Linear(config['TransformerDecoder']['d_model'],
                                          config['FeedForward']['hidden_dims']),
                                nn.Dropout(config['FeedForward']['dropout']),
                                nn.ReLU(),
                                nn.Linear(config['FeedForward']['hidden_dims'],
                                          config['Bert']['num_classes']))
        
    def forward(self,
                antibody_full_tokens,
                antibody_masked_tokens,
                virus_full_tokens,
                cross_attention_mask=None):
        # b_v, s_v, _ = virus_full_tokens.shape
        # b_a, s_a, _ = antibody_full_tokens.shape
        # tgt_mask = torch.zeros((b_a, s_a, s_a), dtype=torch.bool).to(virus_full_tokens.device)
        # memory_mask = torch.zeros((b_a, s_v, s_v), dtype=torch.bool).to(virus_full_tokens.device)
        antibody_features = self.antibody_encoder(**antibody_masked_tokens)['last_hidden_state']
        virus_features = self.virus_encoder(**virus_full_tokens)['last_hidden_state']
        tgt_key_padding_mask = antibody_masked_tokens['attention_mask']
        memory_key_padding_mask = virus_full_tokens['attention_mask']
        # convert int to bool, 0 to True, 1 to False
        tgt_key_padding_mask = tgt_key_padding_mask == 0
        memory_key_padding_mask = memory_key_padding_mask == 0

        cross_features = self.transformer_decoder(antibody_features,
                                                  virus_features,
                                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                                  memory_key_padding_mask=memory_key_padding_mask)
        logits = self.fc(cross_features)
        logits = torch.permute(logits, (0, 2, 1))
        return logits
    
class AutoRegressiveBert(nn.Module):
    '''
    Model class for the model that uses Bert encoder and pytorch transformer decoder layers
    encodes the virus and antibody sequences separately
    Uses the transformer decoder layer to decode the antibody sequence
    Does this in an auto-regressive fashion
    args:
        config: configuration file for the model
    '''
    def __init__(self,
                 config) -> None:
        super(AutoRegressiveBert, self).__init__()
        self.config = config
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=config['TransformerDecoder']['d_model'],
                                                                    nhead=config['TransformerDecoder']['num_heads'],
                                                                    dim_feedforward=config['TransformerDecoder']['dim_feedforward'],
                                                                    batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer,
                                                         num_layers=config['TransformerDecoder']['num_layers'])
        self.virus_and_antibody_encoder = BertModel.from_pretrained(config['Bert']['pretrained_model_name'])
        self.decoder_embedding = BertModel.from_pretrained(config['Bert']['pretrained_model_name']).embeddings

        for key, value in self.virus_and_antibody_encoder.embeddings.named_parameters():
            value.requires_grad = False
        
        for key, value in self.virus_and_antibody_encoder.encoder.named_parameters():
            layer_no = int(key.split('.')[1])
            if layer_no < config['Bert']['start_layer']:
                value.requires_grad = False

        for key, value in self.decoder_embedding.named_parameters():
            value.requires_grad = False

        self.fc = nn.Sequential(nn.Linear(config['TransformerDecoder']['d_model'],
                                          config['FeedForward']['hidden_dims']),
                                nn.Dropout(config['FeedForward']['dropout']),
                                nn.ReLU(),
                                nn.Linear(config['FeedForward']['hidden_dims'],
                                          config['FeedForward']['num_classes']))
        self.max_gen_len = config['AutoRegressive']['max_len_cdrh3'] + 1
        self.sep_token = config['AutoRegressive']['sep_token']
        self.cls_token = config['AutoRegressive']['cls_token']
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self,
                virus_and_antibody_full_tokens):
        b_v = virus_and_antibody_full_tokens['input_ids'].shape[0]
        self.virus_and_antibody_features = self.virus_and_antibody_encoder(**virus_and_antibody_full_tokens)['last_hidden_state']
        memory_key_padding_mask = virus_and_antibody_full_tokens['attention_mask']
        # convert int to bool, 0 to True, 1 to False
        memory_key_padding_mask = memory_key_padding_mask == 0

        out_no = 0
        decoder_input_ids = torch.ones((b_v, 1), 
                                       dtype=torch.long).to(self.virus_and_antibody_features.device) * self.cls_token # N*1
        # out_features = torch.zeros((b_v,
        #                             self.max_gen_len,
        #                             self.config['TransformerDecoder']['d_model'])).to(self.virus_and_antibody_features.device)
        out = torch.zeros((b_v, 
                           self.max_gen_len, 
                           self.config['FeedForward']['num_classes'])).to(self.virus_and_antibody_features.device)
        while out_no < self.max_gen_len:
            decoder_input_embs = self.decoder_embedding(decoder_input_ids) # N*cur_len*d_model
            decoder_input_feats = self.transformer_decoder(decoder_input_embs, # N*cur_len*d_model,
                                                           self.virus_and_antibody_features,
                                                           memory_key_padding_mask=memory_key_padding_mask) # N*cur_len*d_model
            fc_out = self.fc(decoder_input_feats) # N*cur_len*num_classes
            mean_feature = torch.mean(fc_out, dim=1) # N*num_classes
    
            out[:, out_no, :] = mean_feature # N*num_classes
            out_no += 1
            decoder_input_ids = torch.cat((decoder_input_ids,
                                           torch.argmax(mean_feature, dim=-1).unsqueeze(1)), 
                                           dim=1)
        out = torch.permute(out, (0, 2, 1))
        return out

class AutoRegressiveEncoder(nn.Module):
    '''
    Model class for the model that uses Bert encoder
    Used for predicting the masked tokens of CDR
    example: first predict the first CDR position, then use the predicted token to predict the second CDR position
             these positions are masked in the input, and only an encoder is used
    args:
        config: configuration file for the model
    '''
    def __init__(self,
                 config) -> None:
        super(AutoRegressiveEncoder, self).__init__()
        self.config = config
        self.encoder = BertModel.from_pretrained(config['Bert']['pretrained_model_name'])

        for key, value in self.encoder.embeddings.named_parameters():
            value.requires_grad = False

        for key, value in self.encoder.encoder.named_parameters():
            layer_no = int(key.split('.')[1])
            if layer_no < 28:
                value.requires_grad = False
       
        self.fc = nn.Sequential(nn.Linear(config['Bert']['hidden_dims'],
                                          config['FeedForward']['hidden_dims']),
                                nn.ReLU(),
                                nn.Linear(config['FeedForward']['hidden_dims'],
                                          config['Bert']['num_classes']))
        
    def forward(self,
                input_tokens,
                label_tokens,
                mask_start_pos,
                mask_end_pos,
                is_train: bool):
        # input_tokens: N*max_len
        # label_tokens: N*max_len
        # mask_start_pos: 1
        # mask_end_pos: N
        max_len = torch.max(mask_end_pos - mask_start_pos + 1).item()

        return_outs = torch.zeros((input_tokens.shape[0], 
                                   max_len, 
                                   self.config['Bert']['num_classes'])).to(input_tokens.device)

        for i in range(max_len):
            features = self.encoder(**input_tokens)['last_hidden_state']
            logits = self.fc(features) # N*max_len*num_classes

            for b_idx in range(input_tokens.shape[0]):
                return_outs[b_idx, mask_start_pos[b_idx] + i + 1, :] = logits[b_idx, mask_start_pos[b_idx] + i + 1, :]
                if is_train:
                    input_tokens['input_ids'][b_idx, mask_start_pos[b_idx] + i + 1] = label_tokens['input_ids'][b_idx, mask_start_pos[b_idx] + i + 1]
                else:
                    input_tokens['input_ids'][b_idx, mask_start_pos[b_idx] + i + 1] = torch.argmax(logits[b_idx, i, :])

        outs = torch.permute(return_outs, (0, 2, 1))
        return return_outs # N*classes*max_len


# Models for classification into Binds and Non-Binds, Neutralizes and Non-Neutralizes
class ClassificationModel(nn.Module):
    '''
    Model class for the model that uses Bert encoder and a feed forward layer
    Used for classification of Binds and Non-Binds, Neutralizes and Non-Neutralizes
    concatenates the virus and antibody sequence embeddings and passes it through a feed forward layer
    args:
        config: configuration file for the model
    '''
    def __init__(self, 
                 config) -> None:
        super(ClassificationModel, self).__init__()
        self.config = config
        self.encoder = BertModel.from_pretrained(config['Bert']['pretrained_model_name'])

        for key, value in self.encoder.embeddings.named_parameters():
            value.requires_grad = False

        for key, value in self.encoder.encoder.named_parameters():
            layer_no = int(key.split('.')[1])
            if layer_no < config['Bert']['start_layer'] or layer_no > config['Bert']['end_layer']:
                value.requires_grad = False
        self.classifier = nn.Sequential(nn.Linear(config['FeedForward']['input_dims'],
                                                  config['FeedForward']['hidden_dims']),
                                        nn.Dropout(config['FeedForward']['dropout']),
                                        nn.ReLU(),
                                        nn.Linear(config['FeedForward']['hidden_dims'],
                                                  config['Classification']['class_num']))
        
    def forward(self, 
                input_tokens,
                return_features=False):
        # get the features of the cls token
        outputs = self.encoder(**input_tokens)['pooler_output'] # N*hidden_size
        # outputs = outputs[:, 0, :] # N*hidden_size
        if return_features:
            return outputs
        logits = self.classifier(outputs)
        return logits
    
class ClassificationHeirarchicalModel(nn.Module):
    '''
    Model class for the model that uses Bert encoder and a feed forward layer
    Used for classification of Binds and Non-Binds, Neutralizes and Non-Neutralizes
    concatenates the virus and antibody sequence embeddings and passes it through a feed forward layer
    Uses the hierarchical classification approach
    args:
        config: configuration file for the model
    NB: The model is not used, this is for unbalanced data
    '''
    def __init__(self,
                 config) -> None:
        super(ClassificationHeirarchicalModel).__init__()
        self.config = config
        self.encoder = BertModel.from_pretrained(config['Bert']['pretrained_model_name'])
        for key, value in self.encoder.encoder.named_parameters():
            layer_no = int(key.split('.')[1])
            if layer_no < config['Bert']['start_layer'] or layer_no > config['Bert']['end_layer']:
                value.requires_grad = False
        self.feature_extractor = nn.Sequential(nn.Linear(config['FeedForward']['input_dims'],
                                                            config['FeedForward']['hidden_dims']),
                                               nn.Dropout(config['FeedForward']['dropout']),
                                               nn.ReLU())
        self.classifier1 = nn.Sequential(nn.Linear(config['FeedForward']['hidden_dims'],
                                                   4))
        self.excite_layer = nn.Sequential(nn.Linear(1, 32))
        self.classifier2 = nn.Sequential(nn.Linear(32,
                                                   4))
        
    def forward(self,
                input_tokens):
        outputs = self.encoder(**input_tokens)[0]
        outputs = outputs[:, 0, :]
        features = self.feature_extractor(outputs)
        logits1 = self.classifier1(features)
        logits_last = logits1[:, -1].unsqueeze(1)
        logits_last = self.excite_layer(logits_last)
        logits_2 = self.classifier2(logits_last)
        return logits1, logits_2
    
class TorchTransformerClassificationModel(nn.Module):
    '''
    Model class for the model that uses Pytorch's Transformer encoder and a feed forward layer
    Used for classification of Binds and Non-Binds, Neutralizes and Non-Neutralizes
    concatenates the virus and antibody sequence embeddings and passes it through a feed forward layer
    args:
        config: configuration file for the model
    '''
    def __init__(self,
                 config) -> None:
        super(TorchTransformerClassificationModel, self).__init__()
        self.config = config
        # self.tokenizer = BertTokenizer.from_pretrained(config['Bert']['pretrained_model_name'])
        max_len = config['Classification']['max_len'] if not self.config['Classification']['Binds_to_classification']['use_cdrh3'] else 228
        self.pos_embedding = nn.Embedding(max_len + 3,
                                          config['Classification']['d_model'],
                                          padding_idx=0,
                                          )
        self.token_embedding = nn.Embedding(config['Bert']['num_classes'],
                                            config['Classification']['d_model'],
                                            padding_idx=0)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=config['Classification']['d_model'],
                                                                    nhead=config['Classification']['n_heads'],
                                                                    dim_feedforward=config['Classification']['dim_feedforward'],
                                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer,
                                                         num_layers=config['Classification']['n_layers'])
        self.fc = nn.Linear(config['Classification']['d_model'],
                            config['Classification']['class_num'])
        
    def forward(self, input_tokens, return_features=False):
        input_tokens_ids = input_tokens['input_ids'] # N*max_len
        attention_mask = input_tokens['attention_mask'] # N*max_len
        # convert the attention mask to bool, where True means the position is padded
        src_key_padding_mask = attention_mask == 0 # N*max_len
        # convert the attention mask to bool, if not bool
        if src_key_padding_mask.dtype != torch.bool:
            src_key_padding_mask = src_key_padding_mask.bool()

        input_pos_ids = torch.arange(1, input_tokens_ids.shape[1]+1).unsqueeze(0).to(input_tokens_ids.device) # 1*max_len
        input_pos_ids = input_pos_ids.repeat(input_tokens_ids.shape[0], 1) # N*max_len
        input_embs = self.token_embedding(input_tokens_ids) + self.pos_embedding(input_pos_ids) # N*max_len*d_model
        # print(input_embs.shape)
        input_feats =  self.transformer_encoder(input_embs,
                                                src_key_padding_mask=src_key_padding_mask) # N*max_len*d_model 
        mean_feature = torch.mean(input_feats, dim=1) # N*d_model
        if return_features:
            return mean_feature
        logits = self.fc(mean_feature) # N*class_num
        return logits
    
class ClassificationDecoderModel(nn.Module):
    '''
    Model class for the model that uses Bert encoder and a torch multihead attention decoder
    Used for classification of Binds and Non-Binds, Neutralizes and Non-Neutralizes
    Encodes the virus and antibody sequences separately, and applies a multihead attention decoder to the antibody sequence
    args:
        config: configuration file for the model
    '''
    def __init__(self,
                 config) -> None:
        super(ClassificationDecoderModel, self).__init__()
        self.config = config
        self.virus_encoder = BertModel.from_pretrained(config['Bert']['pretrained_model_name'])
        for key, value in self.virus_encoder.named_parameters():
            # print(key)   
            # if 'embedding' in key or 'pooler' in key:
            #     value.requires_grad = False
            #     continue
            value.requires_grad = False
        
        self.antibody_encoder = BertModel.from_pretrained(config['Bert']['pretrained_model_name'])
        for key, value in self.antibody_encoder.embeddings.named_parameters():
            value.requires_grad = False
        for key, value in self.antibody_encoder.encoder.named_parameters():
            # if 'embedding' in key or 'pooler' in key:
            #     value.requires_grad = False
            #     continue
            layer_no = int(key.split('.')[1])
            if layer_no < 29:
                value.requires_grad = False

        self.decoder = nn.MultiheadAttention(config['TransformerDecoder']['d_model'],
                                             config['TransformerDecoder']['num_heads'],
                                             batch_first=True)
        self.fc = nn.Sequential(nn.Linear(config['TransformerDecoder']['d_model'],
                                          config['FeedForward']['hidden_dims']),
                                nn.ReLU(),
                                nn.Linear(config['FeedForward']['hidden_dims'],
                                          2))
        
    def forward(self,
                virus_tokens,
                antibody_tokens):
        # print(anitbody_masked_tokens['input_ids'].shape)
        antibody_features = self.antibody_encoder(**antibody_tokens)['last_hidden_state']
        virus_features = self.virus_encoder(**virus_tokens)['last_hidden_state']

        # virus_features = self.virus_encoder(**virus_full_tokens)['last_hidden_state']
        # cross_attention_mask = torch.zeros((antibody_features.shape[0],
        #                                     antibody_features.shape[1],
        #                                     virus_features.shape[1])).to(antibody_features.device).to(torch.bool)
        
        
        # cross_attention_mask = cross_attention_mask.repeat(self.config['TransformerDecoder']['num_heads'], 
        #                                                    1, 1)
        # # print(antibody_features.shape, virus_features.shape)
        cross_features = self.decoder(antibody_features,
                                      virus_features,
                                      virus_features,
                                      attn_mask=None)[0]
        cross_cls_features = cross_features[:, 0, :]
        logits = self.fc(cross_cls_features)
        return logits

class TransformerWithGraphFeats(nn.Module):
    def __init__(self,
                 config) -> None:
        super(TransformerWithGraphFeats, self).__init__()
        self.config = config
        self.excite_layer = nn.Sequential(nn.Linear(37,
                                                    256),
                                          nn.ReLU(),
                                          nn.Linear(256, 512))
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=config['TransformerDecoder']['d_model'],
                                                        nhead=config['TransformerDecoder']['num_heads'],
                                                        dim_feedforward=config['TransformerDecoder']['dim_feedforward'],
                                                        batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer,
                                             num_layers=config['TransformerDecoder']['num_layers'])
        
        self.fc = nn.Sequential(nn.Linear(config['TransformerDecoder']['d_model'],
                                            config['FeedForward']['hidden_dims']),
                                nn.ReLU(),
                                nn.Linear(config['FeedForward']['hidden_dims'],
                                            2))
        
    def forward(self,
                antibody_input_features,
                virus_input_features,
                antibody_mask,
                virus_mask):
        # antibody_input_features: N*max_len*hidden_dims
        # virus_input_features: N*max_len*hidden_dims

        antibody_input_features = self.excite_layer(antibody_input_features)
        virus_input_features = self.excite_layer(virus_input_features)
        cross_features = self.decoder(antibody_input_features,
                                      virus_input_features,
                                      tgt_key_padding_mask=antibody_mask,
                                      memory_key_padding_mask=virus_mask)
        mean_cross_features = torch.mean(cross_features, dim=1)
        logits = self.fc(mean_cross_features)
        return logits
        
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
    

class AbLangHeavyChainModel(nn.Module):
    def __init__(self,
                 config) -> None:
        super(AbLangHeavyChainModel, self).__init__()
        self.config = config['AbLang']
        self.encoder = AutoModel.from_pretrained(self.config['pretrained_model_name'],
                                                 trust_remote_code=True)
        self.vocab = AutoTokenizer.from_pretrained(self.config['pretrained_model_name']).get_vocab()        
        for key, value in self.encoder.encoder.named_parameters():
            layer_no = int(key.split('.')[1])
            if layer_no < self.config['max_grad_layer']:
                value.requires_grad = False
            print(f"key: {key}, value: {value.requires_grad}")
        # print(f"vocab: {self.vocab}")
        # self.inv_vocab = {v: k for k, v in self.vocab.items()}
        # # sort the inv_vocab by the key, which is the index
        # self.inv_vocab = {k: v for k, v in sorted(self.inv_vocab.items(), key=lambda item: item[0])}
        # print(f"inv_vocab: {self.inv_vocab}")

        self.fc = nn.Sequential(nn.Linear(self.config['Classifier']['input_dims'],
                                          self.config['Classifier']['hidden_dims']),
                                nn.ReLU(),
                                nn.Dropout(self.config['Classifier']['dropout']),
                                nn.Linear(self.config['Classifier']['hidden_dims'],
                                          self.config['Classifier']['num_classes']))
               
    def forward(self,
                input_tokens):
        # input_tokens: {input_ids, attention_mask, token_type_ids}
        encoded_features = self.encoder(**input_tokens)['last_hidden_state']
        logits = self.fc(encoded_features)
        logits = torch.permute(logits, (0, 2, 1))
        return logits # N*classes*max_len

if __name__ ==  "__main__":
    config_file = 'config.yaml'
    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

    # model = ClassificationTwoPositions(config)
    model = AbLangHeavyChainModel(config)
                                        
