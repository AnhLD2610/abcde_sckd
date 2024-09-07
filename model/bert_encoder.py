import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_model import base_model
from transformers import BertModel, BertConfig


class Bert_Encoder(base_model):

    def __init__(self, config):
        super(Bert_Encoder, self).__init__()

        # load model
        self.encoder = BertModel.from_pretrained(config.bert_path).cuda()
        self.bert_config = BertConfig.from_pretrained(config.bert_path)

        # the dimension for the final outputs
        self.output_size = config.encoder_output_size

        self.drop = nn.Dropout(config.drop_out)

        # find which encoding is used
        if config.pattern in ['standard', 'entity_marker']:
            self.pattern = config.pattern
        else:
            raise Exception('Wrong encoding.')
        config.hidden_size = self.bert_config.hidden_size
        config.output_size = config.encoder_output_size
        if self.pattern == 'entity_marker':
            self.encoder.resize_token_embeddings(config.vocab_size + config.marker_size)
            self.linear_transform = nn.Linear(self.bert_config.hidden_size * 2, self.output_size, bias=True)
        else:
            self.linear_transform = nn.Linear(self.bert_config.hidden_size, self.output_size, bias=True)

        self.layer_normalization = nn.LayerNorm([self.output_size])

    def get_output_size(self):
        return self.output_size

    def forward(self, inputs, is_augment = False):
        '''
        :param inputs: of dimension [B, N]
        :return: a result of size [B, H*2] or [B, H], according to different strategy
        '''
        # generate representation under a certain encoding strategy
        if self.pattern == 'standard':
            # in the standard mode, the representation is generated according to
            #  the representation of[CLS] mark.
            output = self.encoder(inputs)[1]
        else:
            if is_augment:
                # Handle the augmented case where two samples are concatenated.
                e11_1 = []
                e21_1 = []
                e11_2 = []
                e21_2 = []
                
                # Get positions of [E11] and [E21] for both concatenated samples in each batch
                for i in range(inputs.size()[0]):
                    tokens = inputs[i].cpu().numpy()
                    e11_1.append(np.argwhere(tokens == 30522)[0][0])  # First sample's E11
                    e21_1.append(np.argwhere(tokens == 30524)[0][0])  # First sample's E21
                    e11_2.append(np.argwhere(tokens == 30522)[1][0])  # Second sample's E11
                    e21_2.append(np.argwhere(tokens == 30524)[1][0])  # Second sample's E21

                # Pass the input to BERT
                tokens_output = self.encoder(inputs)[0]  # [B,N,H]

                output1 = []  # Representations for the first pair of entities (first sequence)
                output2 = []  # Representations for the second pair of entities (second sequence)

                # Extract representations for both [E11] and [E21] for each sample in the batch
                for i in range(len(e11_1)):
                    instance_output = torch.index_select(tokens_output, 0, torch.tensor(i).cuda())
                    
                    # Extract the first pair's representations ([E11] and [E21] from the first sequence)
                    instance_output1 = torch.index_select(instance_output, 1, torch.tensor([e11_1[i], e21_1[i]]).cuda())
                    output1.append(instance_output1)  # [B,2,H]

                    # Extract the second pair's representations ([E11] and [E21] from the second sequence)
                    instance_output2 = torch.index_select(instance_output, 1, torch.tensor([e11_2[i], e21_2[i]]).cuda())
                    output2.append(instance_output2)  # [B,2,H]
                print(output1)
                print(output2)
                print(output1[0].shape)
                # Concatenate the output tensors and reshape them
                output1 = torch.cat(output1, dim=0).view(output1[0].size()[0], -1)  # [B,H*2]
                output2 = torch.cat(output2, dim=0).view(output2[0].size()[0], -1)  # [B,H*2]
                

                return output1, output2  # Return both outputs for the two concatenated samples
            
            else:
                # in the entity_marker mode, the representation is generated from the representations of
                #  marks [E11] and [E21] of the head and tail entities.
                e11 = []
                e21 = []
                # for each sample in the batch, acquire the positions of its [E11] and [E21]
                for i in range(inputs.size()[0]):
                    tokens = inputs[i].cpu().numpy()
                    e11.append(np.argwhere(tokens == 30522)[0][0])
                    e21.append(np.argwhere(tokens == 30524)[0][0])

                # input the sample to BERT
                tokens_output = self.encoder(inputs)[0] # [B,N] --> [B,N,H]
                output = []

                # for each sample in the batch, acquire its representations for [E11] and [E21]
                for i in range(len(e11)):
                    instance_output = torch.index_select(tokens_output, 0, torch.tensor(i).cuda())
                    instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i], e21[i]]).cuda())
                    output.append(instance_output)  # [B,N] --> [B,2,H]

                # for each sample in the batch, concatenate the representations of [E11] and [E21], and reshape
                output = torch.cat(output, dim=0)
                output = output.view(output.size()[0], -1)  # [B,N] --> [B,H*2]

                # the output dimension is [B, H*2], B: batchsize, H: hiddensize
                # output = self.drop(output)
                # output = self.linear_transform(output)
                # output = F.gelu(output)
                # output = self.layer_normalization(output)
            return output