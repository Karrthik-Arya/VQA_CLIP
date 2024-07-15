import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np 
import os
from tqdm import tqdm
import multiprocessing as mp
import json
from data.graph_dataloader import convert_data_format
import torch.nn.functional as F
import pickle
import pandas as pd
import csv
import re
import clip
lab= ["__background__", "airplane", "animal", "arm", "bag", "banana", "basket", "beach", "bear", "bed", "bench", "bike", "bird", "board", "boat", "book", "boot", "bottle", "bowl", "box", "boy", "branch", "building", "bus", "cabinet", "cap", "car", "cat", "chair", "child", "clock", "coat", "counter", "cow", "cup", "curtain", "desk", "dog", "door", "drawer", "ear", "elephant", "engine", "eye", "face", "fence", "finger", "flag", "flower", "food", "fork", "fruit", "giraffe", "girl", "glass", "glove", "guy", "hair", "hand", "handle", "hat", "head", "helmet", "hill", "horse", "house", "jacket", "jean", "kid", "kite", "lady", "lamp", "laptop", "leaf", "leg", "letter", "light", "logo", "man", "men", "motorcycle", "mountain", "mouth", "neck", "nose", "number", "orange", "pant", "paper", "paw", "people", "person", "phone", "pillow", "pizza", "plane", "plant", "plate", "player", "pole", "post", "pot", "racket", "railing", "rock", "roof", "room", "screen", "seat", "sheep", "shelf", "shirt", "shoe", "short", "sidewalk", "sign", "sink", "skateboard", "ski", "skier", "sneaker", "snow", "sock", "stand", "street", "surfboard", "table", "tail", "tie", "tile", "tire", "toilet", "towel", "tower", "track", "train", "tree", "truck", "trunk", "umbrella", "vase", "vegetable", "vehicle", "wave", "wheel", "window", "windshield", "wing", "wire", "woman", "zebra","globalimage"]
with open("/raid/biplab/hassan/VQA_CLIP/embeddings_conceptnet.pkl", "rb") as fIn:
        cache_data = pickle.load(fIn)
        # corpus_sentences = cache_data['words']
        # corpus_embeddings = cache_data['embeddings']
# print(model(["hello"]))
print(cache_data.keys())
def preprocessing(text):
  input_text = text
  input_text = input_text.lower()

  # Removing periods except if it occurs as decimal
  input_text = re.sub(r'(?<!\d)\.(?!\d)', '', input_text)

  # Converting number words to digits
  number_words = {
      "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
      "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
      "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
      "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
      "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
      "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
      "eighty": "80", "ninety": "90"
  }
  input_text = re.sub(r'\b(?:' + '|'.join(number_words.keys()) + r')\b', lambda x: number_words[x.group()], input_text)

  # Removing articles (a, an, the)
  if len(input_text)>3:
    input_text = re.sub(r'\b(?:a|an|the)\b', '', input_text)

  # Adding apostrophe if a contraction is missing it
  input_text = re.sub(r'\b(\w+(?<!e)(?<!a))nt\b', r"\1n't", input_text)

  # input_text = re.sub(r'\b(\w+(?<!t))ent\b', r"\1en't", input_text)

  # Replacing all punctuation (except apostrophe and colon) withinput_text a space character
  input_text = re.sub(r'[^\w\':]|(?<=\d),(?=\d)', ' ', input_text)

  # Removing extra spaces
  input_text = re.sub(r'\s+', ' ', input_text).strip()

  return input_text
maps={}
# for i in range(len(lab)):
#      maps[lab[i]]='NULL'
# for i in range(len(cache_data)):
#      maps[cache_data[i]]=corpus_embeddings[i]
# print(cache_data.keys())
# for i in cache_data.keys():
#      maps[i]=torch.from_numpy(cache_data[i])
for i in cache_data.keys():
     embedding_split = np.array([float(num_str) for num_str in cache_data[i]])
    #  print(embedding_split)
    #  break
     maps[i]=torch.from_numpy(embedding_split)
# with open('./Cnetid-new.csv', 'r') as file:
#     reader = csv.reader(file)
#     datas = list(reader)
# for i in range(len(datas)):
#     if(datas[1][i]=='NULL'):
#          maps[datas[0][i]]='NULL'
def most_common_from_dict(dct):
        lst = [x["answer"] for x in dct]
        return max(set(lst), key=lst.count)
IMAGE_PATH = {
        "train": ("train2014","v2_OpenEnded_mscoco_train2014_questions.json", "v2_mscoco_train2014_annotations.json","train2014"),
        "val": ("val2014","v2_OpenEnded_mscoco_val2014_questions.json", "v2_mscoco_val2014_annotations.json","val2014"),
        "train_ab": ("train2015","OpenEnded_abstract_v002_train2015_questions.json", "abstract_v002_train2015_annotations.json","train2015"),
        "val_ab": ("val2015","OpenEnded_abstract_v002_val2015_questions.json", "abstract_v002_val2015_annotations.json","val2015"),
        "testdev": ("test2015", "v2_OpenEnded_mscoco_test-dev2015_questions.json"),
        "test": ("test2015", "v2_OpenEnded_mscoco_test2015_questions.json"),
        "train_gqa":("images","train_balanced_questions.json"),
        "val_gqa":("images","val_balanced_questions.json"),
        "train_vg":("images","question_answers.json"),
        "val_vg":("images","question_answers.json")}
dataset='VQAv2'
root = 'datasets/vqa_v2'
vocab_path ='vqa_common_ab.txt'
cross='VQAab'
selection = most_common_from_dict
split ='val'
if split[:5]=='train':
    prefix = split[:5]
else: 
    prefix = split[:3]
if dataset=='VQAv2' or dataset=='VQAabs':
        if prefix=='train':
            dir1 = root +'/train/custom_prediction.json'
            dir2 = root +'/train/custom_data_info.json'
            # print("here")
        else: 
            dir1 = root +'/val/custom_prediction.json'
            dir2 = root +'/val/custom_data_info.json'
if dataset=='GQA':
        dir1 = root +'/custom_prediction.json'
        dir2 = root +'/custom_data_info.json'
if dataset=='Super':
        dir1 = root +'/custom_prediction.json'
        dir2 = root +'/custom_data_info.json'
if dataset=='VG':
        dir1 = root +'/custom_prediction.json'
        dir2 = root +'/custom_data_info.json'
f = open(dir1)
data1 = json.load(f)
f1 = open(dir2)
label = json.load(f1)
fps = label["idx_to_files"]
clb = label["ind_to_classes"]
rlb = label["ind_to_predicates"]
if dataset=='VQAv2' or dataset=='VQAabs':
    path = os.path.expanduser(os.path.join(root, IMAGE_PATH[split][1]))
    with open(path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data["questions"])
    # print(df)
    if dataset=="VQAv2":
        df["image_path"] = df["image_id"].apply(
                lambda x: f"{IMAGE_PATH[split][0]}/COCO_{IMAGE_PATH[split][3]}_{x:012d}.jpg")
    elif dataset=="VQAabs":
            df["image_path"] = df["image_id"].apply(
                lambda x: f"{IMAGE_PATH[split][0]}/abstract_v002_{IMAGE_PATH[split][3]}_{x:012d}.png")
    path = os.path.expanduser(os.path.join(root, IMAGE_PATH[split][2]))
    with open(path, 'r') as f:
                data = json.load(f)
    df_annotations = pd.DataFrame(data["annotations"])
    vocab={}
    i=0
    with open(vocab_path, 'r') as file:
        for line in file:
            line = line.replace('\n','')
            vocab[line]=i
            i+=1
    indices=[]
    for i in range(len(df_annotations)):
        selected_answer = preprocessing(selection(df_annotations["answers"][i]))
        # print(selected_answer)
        if selected_answer not in vocab.keys():
            indices.append(i)
    df_annotations.drop(indices,axis=0,inplace=True)
    df_annotations.reset_index(inplace=True,drop=True)
    df = pd.merge(df, df_annotations, left_on='question_id', right_on='question_id', how='right')
    df["image_id"] = df["image_id_x"]
    if not all(df["image_id_y"] == df["image_id_x"]):
                print("There is something wrong with image_id")
    del df["image_id_x"]
    del df["image_id_y"]
    df = df
    n_samples = df.shape[0]
    if split[:5]=='train':
        prefix = split[:5]
    else: 
        prefix = split[:3]
    index =50000 
    # print(n_samples)
    # for i in tqdm(range(n_samples)):
    # # for i in range(50000,min(100000,n_samples)):
    #     image_path = df["image_path"][i]
    #     path_image = os.path.join(root,image_path)
    #     # print(path_image)
    #     question = df["question"][i]
    #     if dataset=='VQAv2' or dataset=='VQAabs':
    #         selected_answers = preprocessing(selection(df["answers"][i]))
    #     else:
    #         selected_answers = preprocessing(df["answer"][i])
    #     ind = fps.index(path_image)
    #     datas = convert_data_format(ind,data1[str(ind)],fps,clb,rlb,maps,question,selected_answers,vocab)
    #     # break
    #     if datas is None:
    #          continue
    #     # print(os.path.join(root, f'processed/{prefix}_{cross}/data_test_{index}.pt'))
    #     torch.save(datas,os.path.join(root, f'processed/{prefix}_{cross}_concept/data_test_{index}.pt'))
    #     # break
    #     index+=126886
    path_image = '/raid/biplab/hassan/datasets/vqa_v2/val2014/COCO_val2014_{:012d}.jpg'.format(131725)
    # path_image = '/raid/biplab/hassan/datasets/vqa_abs/val2015/abstract_v002_val2015_{:012d}.png'.format(26886)
    # path_image = '/raid/biplab/hassan/datasets/vqa_v2/val2014/COCO_val2014_{:012d}.jpg'.format(531299)
    selected_answers = preprocessing('white')
    question = 'What color are the plates?'
    ind = fps.index(path_image)
    datas = convert_data_format(ind,data1[str(ind)],fps,clb,rlb,maps,question,selected_answers,vocab)
    torch.save(datas,os.path.join('/raid/biplab/hassan/VQA_CLIP', 'data_ours_v22.pt'))

    # index =0
elif dataset=='Super':
     path = os.path.expanduser(os.path.join(root, IMAGE_PATH["super"][1]))
     with open(path, 'r') as f:
            data = json.load(f)
     index_tr=0
     index_v=0
     index_t=0
     vocab={}
     i=0
     with open(vocab_path, 'r') as file:
        for line in file:
            line = line.replace('\n','')
            vocab[line]=i
            i+=1
     for a in tqdm(data['questions']):
          image_id = a['image_filename']
          image_path = os.path.expanduser(os.path.join(root, IMAGE_PATH["super"][0]))
          path_image = os.path.join(image_path,image_id)
          question = a['question']
          selected_answers = (str)(a['answer'])
          ind = fps.index(path_image)
          id = a['image'].split('_')[2]
          datas = convert_data_format(ind,data1[str(ind)],fps,clb,rlb,maps,question,selected_answers,vocab)
          if datas is None:
             continue
          break
          if (int)(id)>=25000:
            torch.save(datas,os.path.join(root, f'processed/test/data_test_{index_t}.pt'))
            index_t+=1
          elif (int)(id)>=20000:
            torch.save(datas,os.path.join(root, f'processed/val/data_test_{index_v}.pt'))
            index_v+=1
          else:
            torch.save(datas,os.path.join(root, f'processed/train/data_test_{index_tr}.pt'))
            index_tr+=1
          # break


elif dataset=='VG':
         vocab={}
         i=0
         with open(vocab_path, 'r') as file:
            for line in file:
                vocab[line[:-1]]=i
                i+=1
         path = os.path.expanduser(os.path.join(root, IMAGE_PATH[split][1]))
         with open(path, 'r') as f:
            data = json.load(f)
         leng = len(data)
        #  print(leng)
         counts=0
         i=0
         index=0
         all_embeddings = []
         all_captions = []
         for answer in tqdm(data):
            if split[:5] == 'train' and counts<(int)(leng*0.8):
                for q in answer['qas']:
                    d={}
                    question = q["question"]
                    selected_answers = preprocessing(q["answer"])
                    if selected_answers in vocab.keys():
                        path = os.path.join(IMAGE_PATH[split][0],str(q["image_id"])+".jpg")
                        path = os.path.join(root,path)
                        ind = fps.index(path)
                        datas = convert_data_format(ind,data1[str(ind)],fps,clb,rlb,maps,question,selected_answers,vocab)
                        if datas is None:
                            continue
                        torch.save(datas,os.path.join(root, f'processed/{prefix}_{cross}/data_test_{index}.pt'))
                        index+=1
                        
            elif split[:3] == 'val' and counts>=(int)(leng*0.8):
                    for q in answer['qas']:
                        d={}
                        question = q["question"]
                        selected_answers = preprocessing(q["answer"])
                        if selected_answers in vocab.keys():
                            path = os.path.join(IMAGE_PATH[split][0],str(q["image_id"])+".jpg")
                            path = os.path.join(root,path)
                            ind = fps.index(path)
                            datas = convert_data_format(ind,data1[str(ind)],fps,clb,rlb,maps,question,selected_answers,vocab)
                            if datas is None:
                                continue
                            torch.save(datas,os.path.join(root, f'processed/{prefix}_{cross}/data_test_{index}.pt'))
                            index+=1
            counts+=1
else:
    vocab={}
    i=0
    with open(vocab_path, 'r') as file:
        for line in file:
            vocab[line[:-1]]=i
            i+=1 
    path = os.path.expanduser(os.path.join(root, IMAGE_PATH[split][1]))
    with open(path, 'r') as f:
        datas = json.load(f)
    index=0
    i=0
    path_image = '/raid/biplab/hassan/datasets/gqa/images/2361083.jpg'
    # path_image = '/raid/biplab/hassan/datasets/vqa_v2/val2014/COCO_val2014_{:012d}.jpg'.format(531299)
    selected_answers = preprocessing('walking')
    question = 'What is the person to the left of the car doing?'
    ind = fps.index(path_image)
    datas = convert_data_format(ind,data1[str(ind)],fps,clb,rlb,maps,question,selected_answers,vocab)
    torch.save(datas,os.path.join('/raid/biplab/hassan/VQA_CLIP', 'data_ours_gqa.pt'))

    # for answer in tqdm(datas.values()):
    #         question = answer["question"]
    #         d={}
    #         selected_answers = preprocessing(answer["answer"])
    #         if selected_answers in vocab.keys():
    #             path = os.path.join(IMAGE_PATH[split][0],answer["imageId"]+".jpg")
    #             path = os.path.join(root,path)
    #             ind = fps.index(path)
    #             datas = convert_data_format(ind,data1[str(ind)],fps,clb,rlb,maps,question,selected_answers,vocab)
    #             torch.save(datas,os.path.join(root, f'processed/{prefix}_{cross}/data_test_{index}.pt'))
    #             index+=1
            