from tqdm import tqdm
import numpy as np
import _pickle

dict_embedding = {}
# lab= ["airplane", "animal", "arm", "bag", "banana", "basket", "beach", "bear", "bed", "bench", "bike", "bird", "board", "boat", "book", "boot", "bottle", "bowl", "box", "boy", "branch", "building", "bus", "cabinet", "cap", "car", "cat", "chair", "child", "clock", "coat", "counter", "cow", "cup", "curtain", "desk", "dog", "door", "drawer", "ear", "elephant", "engine", "eye", "face", "fence", "finger", "flag", "flower", "food", "fork", "fruit", "giraffe", "girl", "glass", "glove", "guy", "hair", "hand", "handle", "hat", "head", "helmet", "hill", "horse", "house", "jacket", "jean", "kid", "kite", "lady", "lamp", "laptop", "leaf", "leg", "letter", "light", "logo", "man", "men", "motorcycle", "mountain", "mouth", "neck", "nose", "number", "orange", "pant", "paper", "paw", "people", "person", "phone", "pillow", "pizza", "plane", "plant", "plate", "player", "pole", "post", "pot", "racket", "railing", "rock", "roof", "room", "screen", "seat", "sheep", "shelf", "shirt", "shoe", "short", "sidewalk", "sign", "sink", "skateboard", "ski", "skier", "sneaker", "snow", "sock", "stand", "street", "surfboard", "table", "tail", "tie", "tile", "tire", "toilet", "towel", "tower", "track", "train", "tree", "truck", "trunk", "umbrella", "vase", "vegetable", "vehicle", "wave", "wheel", "window", "windshield", "wing", "wire", "woman", "zebra"]
relations =  ["above", "across", "against", "along", "and", "at", "attached to", "behind", "belonging to", "between", "carrying", "covered in", "covering", "eating", "flying in", "for", "from", "growing on", "hanging from", "has", "holding", "in", "in front of", "laying on", "looking at", "lying on", "made of", "mounted on", "near", "of", "on", "on back of", "over", "painted on", "parked on", "part of", "playing", "riding", "says", "sitting on", "standing on", "to", "under", "using", "walking in", "walking on", "watching", "wearing", "wears", "with"]
with open("/raid/biplab/hassan/comp_log_dot_0.1.tsv", "r") as raw_file:
        for entry in tqdm(raw_file, desc="Saving the node embeddings"):
            entry.strip()
            entry = entry.replace("\n", "")
            word = entry.split("\t")[0]
            if word and len(word.split("/"))>2 and word.split("/")[-2]=='en':
                embedd = entry.split("\t")[1:]
                # print(embedd)
                embedding_split = [float(num_str) for num_str in embedd]
                # print()

                # embedding_split = entry.replace(" \n", "").split(" ")
                words = word.split("/")[-1]
                # print(words)
                # print(embedding_split)
                words = words.replace("_", " ")
                if words in relations:
                    embedding = np.asarray(embedding_split)
                    dict_embedding[words] = embedding
                # print(word)
            # break
            # if word in lab:
            #         embedding = np.asarray(embedding_split[1:])
            #         dict_embedding[word] = embedding

print(dict_embedding.keys())
print(len(dict_embedding))
print(len(relations))
for k in relations:
     if k not in dict_embedding.keys():
          print(k)
if len(dict_embedding.keys())==len(relations):
     with open(
        "/raid/biplab/hassan/VQA_CLIP/embeddings_cskg_rel.pkl", "wb"
    ) as pkl_file:
        _pickle.dump(dict_embedding, pkl_file)
     print("All keys matched !!! and saved")