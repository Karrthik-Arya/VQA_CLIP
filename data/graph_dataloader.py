import torch
from torch_geometric.data import Data
import json
import clip
import numpy as np
import cv2
from PIL import Image,ImageDraw
import torch.nn.functional as F
import pickle
import csv
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
def get_size(image_size):
    min_size = 600
    max_size = 1000
    w, h = image_size
    size = min_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))
    if (w <= h and w == size) or (h <= w and h == size):
        return (w, h)
    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    return (ow, oh)
def draw_single_box(pic, box, color='red', draw_info=None):
    draw = ImageDraw.Draw(pic)
    x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline=color)
    if draw_info:
        draw.rectangle(((x1, y1), (x1+50, y1+10)), fill=color)
        info = draw_info
        draw.text((x1, y1), info)
def convert_data_format(id,data,fps,clb,rlb,maps,question,selected_answers,vocab):
    node_features = []
    edge_index = []
    edge_attributes = []
    node_labels =[]
    # print(id)
    # Iterate over the objects
    nodeid=0
    # img = cv2.imread(fps[id])
    with torch.no_grad():
        size = get_size(Image.open(fps[id]).size)
        # print(fps[id])
        img = Image.open(fps[id]).resize(size)
        flag=0
        image = preprocess(img).unsqueeze(0).to(device)
        global_features = model.encode_image(image)
        # node_features.append(global_features)
        s = maps["zebra"]
        global_feats = torch.zeros_like(s)
        crops=[]
        nodes=[]
        for obj_data in data['bbox']:
            # Extract node features
            # print(clip.tokenize(obj_data['name']).shape)
            x1= (int)(obj_data[0])
            y1 = (int)(obj_data[1])
            x2 = (int)(obj_data[2])
            y2 = (int)(obj_data[3])
            area = (x1, y1, x2, y2)
            # print(area)
            # print(fps[id])
            # print(clb[data['bbox_labels'][nodeid]])
            # Crop, show, and save image
            # print(x1,y1,x2,y2)
            # print(img.size)
            cropped_img = img.crop(area)
            # draw_single_box(img,obj_data, color='red', draw_info=clb[data['bbox_labels'][nodeid]])
            # if flag==0:
                #  img.save('./test.png')
                #  print(clb[data['bbox_labels'][nodeid]])
            # flag=1
            image = preprocess(cropped_img).unsqueeze(0).to(device)
            crops.append(image)
            nodes.append(nodeid)
            nodeid+=1
        if(len(crops)>0):
            crops = torch.stack(crops).squeeze(1)
            with torch.no_grad():
                image_features = model.encode_image(crops)
        for i in range(len(nodes)):
            s = maps[clb[data['bbox_labels'][nodes[i]]]]
            node_labels.append(clb[data['bbox_labels'][nodes[i]]]) ###change here
            # global_feats+=s
            # if  s!= 'NULL':
            #     s = s.to('cuda')
            # node_features.append(image_features)
            #     #  print(torch.stack(node_features).shape)

            # else:

            s = s.to('cuda').unsqueeze(0)
            
            # print(torch.cat((image_features,s),dim=1).shape)
            node_features.append(torch.cat((image_features[i].unsqueeze(0),s),dim=1))
            
            # inputs = clip.tokenize().to(device)
            # text_features = model.get_text_features(**inputs)
            # print(inputs.shape)
            # )
            # print(text_features)
            # print(text_features.shape)
            
            # nodeid+=1

        edgeid=0
        # img.save('./test.png')
        # Extract relations
        edges =[]
        for relation in data['rel_pairs']:
                # target_obj_id = relation['object']
                # target_obj_name = relation['name']
                # inputs = tokenizer([target_obj_name], padding=True, return_tensors="pt")
                edge_att = rlb[data["rel_labels"][edgeid]]
                edge_index.append([relation[0], relation[1]])
                # edge_attributes.append(edge_features)
                edges.append(edge_att)
                edgeid+=1
        # edges = torch.stack(edges)
        inputs = clip.tokenize(edges).to(device)
        edge_features = model.encode_text(inputs)
        # edge_attributes.append(edge_features)
        for i in range(len(edges)):
            edge_attributes.append(edge_features[i].unsqueeze(0))
        # s = maps["globalimage"]
        # s = s.to('cuda').unsqueeze(0)
        # print(torch.cat((global_features,s),dim=1).shape)
        # global_feats = F.normalize(global_feats , p=2.0, dim = 0)
        # global_feats = global_feats.to('cuda').unsqueeze(0)
        # node_features.append(torch.cat((global_features,global_feats),dim=1))
        # # node_features.append(global_features)
        # inputs = clip.tokenize("Global features").to(device)
        # edge_features = model.encode_text(inputs)
        # edge_index.append([nodeid,nodeid])
        # edge_attributes.append(edge_features)
        # inputs = clip.tokenize("Patch of").to(device)
        # edge_features = model.encode_text(inputs)
        # for i in range(len(node_features)-1):
        #     edge_index.append([i,nodeid])
        #     edge_attributes.append(edge_features)
        if len(node_features)==0 or len(edge_features)==0:
            return None
        node_features = torch.stack(node_features)
        node_features = node_features.squeeze(1)
        edge_attributes = torch.stack(edge_attributes)
        edge_attributes = edge_attributes.squeeze(1)
        edge_index = torch.tensor(edge_index).t().contiguous()
            


        # Create a PyTorch Geometric Data object
        # if len(node_features)==0 or len(edge_features)==0 or len(edge_attributes)==0:
        #      print("Unfortunate")
        #      print(id)
        question = clip.tokenize(question)
        # question = model.encode_text(question)
        # img variation 114
        # answer = F.one_hot(torch.tensor(vocab[selected_answers]), num_classes=1000).to(torch.float32)
        answer = torch.tensor(vocab[selected_answers])
        # node_features = node_features.to('cpu')
        # edge_index = edge_index.to('cpu')
        # edge_attributes = edge_attributes.to('cpu')
        # question = question.to('cpu')
        # answer = answer.to('cpu')
        # global_features = global_features.to('cpu')
        graph_data = Data(x=node_features,labels=node_labels,global_feature=global_features, edge_index=edge_index,edge_attr=edge_attributes,question=question,answer=answer).to('cpu')
        # print(id)
    return graph_data