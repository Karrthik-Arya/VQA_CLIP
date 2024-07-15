import torch
from torch_geometric.data import Dataset, Data
import os
def count_files(directory):
    file_list = os.listdir(directory)
    total_files = len(file_list)
    return total_files
def convert_to_tensor(data):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    return data
class GQADataset(Dataset):
    def __init__(self, root,split,dataset,cross,test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.dataset= dataset
        self.root = root
        self.test = test
        self.split = split
        self.cross = cross
        self.filename= "None"
        super(GQADataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        if self.split[:5]=='train':
            part = 'train'
        else:
            part='val'
        # directory = self.root +'/processed/'+ part +"_" + self.cross+""
        directory = self.root +'/processed/'+ part +"_" + self.cross +"_concept"
        count=0
        for filename in os.listdir(directory):
            if filename.endswith(".pt"):
                count += 1
        if part =='train':
            self.n_samples = count
        else:
            self.n_samples = count
            
        # print("Found: ",self.n_samples)
        if self.split[:5]=='train':
            # return [f'{part}_{self.cross}/data_test_{i}.pt' for i in range(self.n_samples)]
            return [f'{part}_{self.cross}_concept/data_test_{i}.pt' for i in range(self.n_samples)]
        else: 
            # return [f'{part}_{self.cross}/data_test_{i}.pt' for i in range(self.n_samples)]
            return [f'{part}_{self.cross}_concept/data_test_{i}.pt' for i in range(self.n_samples)]
    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        # return self.n_samples
        # if self.split=='train_ab':
        #     return 15000
        # elif self.split=='train':
        #     return 100000
        # else:
        return self.n_samples

    def get(self, idx):
        if self.split[:5]=='train':
            part = 'train'
        else:
            part='val'
        if self.split=='train' or self.split=='val' or self.split=='train_ab' or self.split=='val_ab':
            idx+=50000
        data = torch.load(os.path.join(self.processed_dir, f'{part}_{self.cross}_concept/data_test_{idx}.pt'))
        # data1 = torch.load(os.path.join(self.processed_dir, f'{part}_{self.cross}_cskg/data_test_{idx}.pt'))
        # print(data.x.shape)
        # print(data1.x.shape)
        # data.x  = torch.cat((data.x,data1.x[:,512:]),dim=1)
        data.edge_index = convert_to_tensor(data.edge_index)
        data.edge_attr  = convert_to_tensor(data.edge_attr)
        data.answer = data.answer.to(torch.long)
        # print(data.answer.shape)
        # print(data.edge_index.shape)
        return data