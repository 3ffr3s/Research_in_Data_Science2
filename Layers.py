import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np 

class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self,q,k,v, mask = None, dropout = None):

        # q,k,v shape : (batch_size,seq_len, d_k or d_v)
        score = torch.matmul(q, k.transpose(-1,-2)) / np.sqrt(self.d_k)
       
        if mask is not None:
            mask = torch.as_tensor(mask, dtype = torch.bool)
            score.masked_fill_(mask, -1e9)

        score = F.softmax(score, dim = -1)
        if dropout is not None:
            score = F.dropout(score, p = dropout)
            
        output = torch.matmul(score,v)
        
        return score, output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model,num_heads, d_k, d_v, dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v

        self.WQ = nn.Linear(self.d_model, self.d_k * self.num_heads, bias= False)
        self.WK = nn.Linear(self.d_model, self.d_k * self.num_heads, bias= False) 
        self.WV = nn.Linear(self.d_model, self.d_v * self.num_heads, bias= False)
        self.linear = nn.Linear(self.d_v * self.num_heads ,self.d_model, bias = False)
        
        self.ScaledDotProductAttention = ScaledDotProductAttention(self.d_k)
        
        self.dropout = nn.Dropout(p = dropout) 
        self.layer_norm = nn.LayerNorm(normalized_shape = d_model, eps=1e-6)  
        
    def forward(self,input,mask = None):
        # input_shape : (batch_size, seq_len, d_model)
        batch_size = input.shape[0]
        seq_len = input.shape[1]

        residual = input
        
        WQ = self.WQ(input).view(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2) 
        WK = self.WK(input).view(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)
        WV = self.WV(input).view(batch_size,seq_len,self.num_heads,self.d_v).transpose(1,2)
        # WQ, WK, WV shape : (batch_size, num_heads, seq_len, d_k or d_v)

        if mask is not None:   # mask dimension : (batch_size, seq_len, seq_len)
            mask = mask.unsqueeze(1)

        score, context = self.ScaledDotProductAttention(WQ,WK,WV, mask)  #context shape : (batch_size, num_heads, seq_len, d_v)
        context = context.transpose(1,2).contiguous().view(batch_size,seq_len, self.num_heads * self.d_v)
        output = self.linear(context)  # output shape : (batch_size, seq_len, d_model)
        output = self.dropout(output)
        
        output = self.layer_norm(output + residual)
        
        return score, output 

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_hid, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_hid = d_hid
        
        self.linear1 = nn.Linear(self.d_model, self.d_hid)
        self.linear2 = nn.Linear(self.d_hid, self.d_model)
        
        self.dropout = nn.Dropout(p = dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape = self.d_model, eps=1e-6)
    
    def forward(self,input):
        residual = input
        
        output = F.relu(self.linear1(input))
        output = self.linear2(output)
        output = self.dropout(output)
        output = self.layer_norm(output+residual)
        
        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self,seq_len,d_model, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        pe = torch.zeros((self.seq_len, self.d_model), requires_grad = False)
        self.dropout = nn.Dropout(p =dropout)

        for pos in range(self.seq_len):
            for i in range(self.d_model):
                if i%2 == 0:
                    pe[pos][i] = math.sin(pos/(math.pow(10000,i/self.d_model)))
                elif i%2 ==1:
                    pe[pos][i] = math.cos(pos/(math.pow(10000,(i-1)/self.d_model)))
                    
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self,input):
        return self.dropout(input + self.pe)

class AdditiveAttention(nn.Module):
    def __init__(self,hid_dim,attn_dim):
        super().__init__()
        
        self.attn1 = nn.Linear(hid_dim, attn_dim)
        self.attn2 = nn.Linear(attn_dim,1, bias = False)
        self.softmax = nn.Softmax()

    def forward(self,input):
        attn1 = torch.tanh(self.attn1(input))   # attn1 shape : (batch_size, seq_len, attn_dim)
        attn2 = self.attn2(attn1).squeeze(-1)               # attn2 shape : (batch_size, seq_len, 1)
        attn_score = self.softmax(attn2).unsqueeze(-1)
        output = (attn_score * input).sum(dim = 1)
        return attn_score, output

class ByteEncoderLayer(nn.Module):
    def __init__(self,d_model,num_heads, d_k, d_v, d_hid, dropout = 0.1):
        super().__init__()
        self.MultiHeadAttention = MultiHeadAttention(d_model, num_heads, d_k, d_v,dropout = dropout)
        self.PositionwiseFeedforward = PositionwiseFeedforward(d_model, d_hid, dropout = dropout)
        
    def forward(self,input, mask = None):
        score, output = self.MultiHeadAttention(input,mask = mask)
        output = self.PositionwiseFeedforward(output)
        return score, output
    
class ByteEncoder(nn.Module):
    def __init__(self,num_layers, d_model,num_heads, d_k, d_v, d_hid, dropout = 0.1):
        super().__init__()
        self.encoder = nn.ModuleList([ByteEncoderLayer(d_model,num_heads, d_k, d_v, d_hid, dropout = dropout) for _ in range(num_layers)])
        
    def forward(self,input):
        output = input
        for layer in self.encoder:
            score, output = layer(output)
        return score, output
    
class Flow_CLF(nn.Module):
    def __init__(self, num_layers, d_model,num_heads, d_k, d_v, d_hid, add_attn_dim, pck_len,device, num_classes = 2, dropout = 0.1):
        super().__init__()
        
        self.device = device
        self.d_model = d_model
        
        self.byte_emb = nn.Embedding(256, d_model)
        self.positionalEncoder =  PositionalEncoding(pck_len,d_model, dropout = dropout)     #seq_len 이거 다시 설정해줘야 함 
        
        #Byte
        self.ByteEncoder = ByteEncoder(num_layers, d_model,num_heads, d_k, d_v, d_hid, dropout = dropout)
        
        #Packet
        self.PacketEncoder = AdditiveAttention(d_model,add_attn_dim)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self,input): 
        batch_size = input.shape[0]
        flow_len = input.shape[1]
        
        byte_embeddings = self.byte_emb(input)
        byte_embeddings = self.positionalEncoder(byte_embeddings)  # output shape : (batch_size,flow_len,pck_len,d_model)
        
        packet_embeddings = torch.zeros([batch_size,flow_len,self.d_model]).to(self.device)
        
        for idx, flow in enumerate(byte_embeddings):
           
            score, byte_encoding = self.ByteEncoder(flow)              
            packet_embedding = byte_encoding.mean(dim = 1)
            packet_embeddings[idx,:,:] = packet_embedding
        
        score, output = self.PacketEncoder(packet_embeddings)
        output = self.fc(output)
        
        return output
