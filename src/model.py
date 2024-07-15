from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GAT,BatchNorm,GATConv
from torch_geometric.nn.pool import global_mean_pool
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import math
import clip

def make_mask(feature):
    # return (torch.sum(
    #     torch.abs(feature),
    #     dim=-1
    # ) == 0).unsqueeze(1).unsqueeze(2)
    return ~(feature == 0).unsqueeze(1).unsqueeze(2)
class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x
class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.hidden_size, __C.hidden_size)
        self.linear_k = nn.Linear(__C.hidden_size, __C.hidden_size)
        self.linear_q = nn.Linear(__C.hidden_size, __C.hidden_size)
        self.linear_merge = nn.Linear(__C.hidden_size, __C.hidden_size)

        self.dropout = nn.Dropout(__C.dropout_r)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.multi_head,
            int(self.__C.hidden_size / self.__C.multi_head)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.multi_head,
            int(self.__C.hidden_size / self.__C.multi_head)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.multi_head,
            int(self.__C.hidden_size / self.__C.multi_head)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)
        if mask.dim() > 4:
            mask = mask.squeeze(4)
        # mask = mask.squeeze()
        # print(mask.shape)
        # print(scores.shape)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)
    
class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))
    
class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.hidden_size,
            mid_size=__C.ff_size,
            out_size=__C.hidden_size,
            dropout_r=__C.dropout_r,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)

class Encoder(nn.Module):
    def __init__(self, __C):
        super(Encoder, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.dropout_r)
        self.norm1 = LayerNorm(__C.hidden_size)

        self.dropout2 = nn.Dropout(__C.dropout_r)
        self.norm2 = LayerNorm(__C.hidden_size)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))
        return y

class Decoder(nn.Module):
    def __init__(self, __C):
        super(Decoder, self).__init__()
        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.dropout_r)
        self.norm1 = LayerNorm(__C.hidden_size)

        self.dropout2 = nn.Dropout(__C.dropout_r)
        self.norm2 = LayerNorm(__C.hidden_size)

        self.dropout3 = nn.Dropout(__C.dropout_r)
        self.norm3 = LayerNorm(__C.hidden_size)

    def forward(self, x, y, x_masks, y_mask):

        x = self.norm2(x + self.dropout2(
            self.mhatt1(v=x, k=x, q=x, mask=x_masks)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C        
        self.mlp = MLP(
            in_size=__C.hidden_size,
            mid_size=__C.flat_mlp_size,
            out_size=__C.flat_glimpses,
            dropout_r=__C.dropout_r,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.hidden_size * __C.flat_glimpses,
            __C.flat_out_size
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        # print(x_mask.shape)
        # print(att.shape)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
class SceneGraphformerattention(nn.Module):
    def __init__(self,__C,n_class):
        super(SceneGraphformerattention, self).__init__()
        # self.conv1 = GAT(-1,__C.hidden_size,__C.gat_layers,dropout=__C.gat_dropout)
        self.conv1 = GATConv(-1,__C.hidden_size,__C.gat_layers,dropout=__C.gat_dropout)
        self.conv2 = GATConv(__C.hidden_size,__C.hidden_size,__C.gat_layers,dropout=__C.gat_dropout)
        # self.conv2 = GAT(-1,__C.hidden_size,__C.gat_layers,dropout=__C.gat_dropout)
        self.bn1 = BatchNorm(__C.hidden_size)
        self.relu = nn.ReLU()
        # self.ln1 = nn.LayerNorm(128)
        # self.projection = nn.Linear(712,512)
        # self.lstm = nn.GRU(
        #     input_size= 512,
        #     hidden_size=__C.hidden_size,
        #     num_layers=__C.lstm_layer,
        #     batch_first=True
        # )
        self.enc_list = nn.ModuleList([Encoder(__C) for _ in range(__C.layer)])
        self.dec_list = nn.ModuleList([Decoder(__C) for _ in range(__C.layer)])
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)
        # self.pool1 = nn.AdaptiveAvgPool1d(1)
        # self.pool2 = nn.AdaptiveAvgPool1d(1)
        # __C.flat_out_size
        self.proj_norm = LayerNorm(1024) # 1024
        self.proj = nn.Linear(1024,n_class) # 1024
    def permute(self,x,batch):

        for i in range(len(batch)):
            leng = torch.sum(batch[i]).item()
            permutation = torch.randperm(leng)
            x[i][:leng] = x[i][permutation]
        return x


    def forward(self, node_features, edge_index,ques,graph_labels,ques_token,img):
        # x = node_features[:,:512]
        # print(node_features.shape)
        #[:,:512]
        x = node_features
        # print(x.shape)
        # x = node_features
        # x = torch.cat((embed1,embed2),dim=1)
        ques = ques.squeeze(1)
        ques_token = ques_token.squeeze(1)
        # print("Actual shape",ques_token.shape)
        y_mask = ~(make_mask(ques_token))
        # y, _ = self.lstm(ques.squeeze(1))
        # print(y_mask)
        num_trues = torch.sum(y_mask).item()
        # print(y_mask[0,0,0,:])
        # print(f"Number of True values: {num_trues}")
        y = ques.clone()
        
        x,w = self.conv1(x, edge_index,return_attention_weights=True)
        # x,w = self.conv2(x, edge_index,return_attention_weights=True)
        x = self.relu(self.bn1(x))
        # x1 = self.relu(self.bn1(self.conv2(x1, edge_index)))
        # x = torch.cat((x,x1),dim=1)
        # sums =  torch.
        # x = self.projection(x)
        x, batch = to_dense_batch(x,graph_labels,max_num_nodes=ques.size(1))
        x1 = x.clone()
        # x1 = self.permute(x1,batch)
        # x1, batch = to_dense_batch(x1,graph_labels,max_num_nodes=ques.size(1))
        # x_mask = make_mask(batch)
        x_mask = (~batch).unsqueeze(1).unsqueeze(2)
        # print(x_mask)
        # print(x_mask.shapec)
        # ques = model.encode_text(ques_token)
        # x = global_mean_pool(x,graph_labels)
        for dec,enc in zip(self.dec_list,self.enc_list):
            # y1 = enc(y, y_mask)
            # y1 = enc(y1, y_mask)
            y1 = y
            x = dec(x,y1, x_mask,y_mask)
            # x = dec(x,y1, x_mask,y_mask)
            y=y1

        x_t = self.attflat_img(
            x,
            x_mask
        )
        y=ques.clone()
        # with torch.no_grad():
        #     for dec,enc in zip(self.dec_list,self.enc_list):
        #         y1 = enc(y, y_mask)
        #         x1 = dec(x1,y1, x_mask,y_mask)
        #         y=y1

        #     x1 = self.attflat_img(
        #         x1,
        #         x_mask
        #     )

        x = self.proj_norm(x_t)
        # print(ques_token.shape)
        # proj_feat = torch.cat((ques,x),dim=1)
        out = self.proj(x)
    
        return out,x_t,x1,w
    
class SceneGraph(nn.Module):
    def __init__(self,__C,n_class):
        super(SceneGraph, self).__init__()
        self.conv1 = GAT(-1,__C.hidden_size,__C.gat_layers,dropout=__C.gat_dropout)
        self.conv2 = GAT(-1,__C.hidden_size,__C.gat_layers,dropout=__C.gat_dropout)
        self.bn1 = BatchNorm(__C.hidden_size)
        self.relu = nn.ReLU()
        self.ln1 = nn.LayerNorm(128)
        self.lstm = nn.GRU(
            input_size= 512,
            hidden_size=__C.hidden_size,
            num_layers=__C.lstm_layer,
            batch_first=True
        )
        self.enc_list = nn.ModuleList([Encoder(__C) for _ in range(__C.layer)])
        self.dec_list = nn.ModuleList([Decoder(__C) for _ in range(__C.layer)])
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)
        self.pool1 = nn.AdaptiveAvgPool1d(1)
        self.pool2 = nn.AdaptiveAvgPool1d(1)
        # __C.flat_out_size
        self.proj_norm = LayerNorm(1024) # 1024
        self.proj = nn.Linear(1024,n_class) # 1024

    def forward(self, node_features, edge_index,ques,graph_labels,ques_token,img):
        x = node_features
        ques_token = ques_token.squeeze(1)
        x = self.relu(self.bn1(self.conv1(x, edge_index)))
        ques = model.encode_text(ques_token)
        x = global_mean_pool(x,graph_labels)
        proj_feat = torch.cat((ques,x),dim=1)
        out = self.proj(proj_feat)
    
        return out