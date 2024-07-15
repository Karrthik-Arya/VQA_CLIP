import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from data.dataset import GQADataset
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from src.model import SceneGraphformerattention,SceneGraph
import clip
import numpy as np
import torch.nn as nn
import random
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch (might reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
model1, preprocess = clip.load("ViT-B/32",device='cuda',jit=False)
maps = {"VQAv2" : "/raid/biplab/hassan/datasets/vqa_v2","VQAab": "/raid/biplab/hassan/datasets/vqa_abs","VG": "/raid/biplab/hassan/datasets/vg","GQA": "/raid/biplab/hassan/datasets/gqa"}
@hydra.main(config_path="conf",config_name="config")
def train(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    # Initialize model
    set_seed(cfg.data.seed) 

    # Initialize optimizer with tuned hyperparameters
    print("lr :",cfg.data.lr)
    print("seed: ",cfg.data.seed)
    print("batch_size ",cfg.data.batch_size)
    print("not using commonsense: ",cfg.model.ncs)
    print("attention: ",cfg.model.attention)
    print("from {} to {}".format(cfg.data.path,cfg.data.path_target))

    if cfg.model.attention:
        model = SceneGraphformerattention(cfg.model,cfg.model.n_class).to('cuda')
    else:
        model = SceneGraph(cfg.model,cfg.model.n_class).to('cuda')
    optimizer = optim.SGD(model.parameters(), lr=cfg.data.lr)

    train_dataset = GQADataset(maps[cfg.data.path],'train',cfg.data.path,cfg.data.cross)
    val_dataset = GQADataset(maps[cfg.data.path],'val',cfg.data.path,cfg.data.cross)
    cross_dataset = GQADataset(maps[cfg.data.path_target],'val_ab',cfg.data.path_target,cfg.data.cross)

    print(len(train_dataset))
    print(len(val_dataset))
    print(len(cross_dataset))

    # train_module = LightningDataset(train_dataset,val_dataset)
    loss_mse = nn.MSELoss()

    train_dataloader = DataLoader(train_dataset,batch_size=cfg.data.batch_size,num_workers=4) # 
    val_dataloader = DataLoader(val_dataset,batch_size=cfg.data.batch_size,num_workers=4)
    cross_dataloader = DataLoader(cross_dataset,batch_size=cfg.data.batch_size,num_workers=4)

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.data.num_epochs)

    best_accuracy = 0
    cross_best = 0
    for epoch in range(cfg.data.num_epochs):
        model.train()
        total_loss = 0.0
        correct_predictions=0
        total_samples=0
        for batch in train_dataloader:
            inputs= batch.to('cuda')
            # print(inputs.question.shape)
            if cfg.model.ncs:
                inputs.x = inputs.x[:,:512]
            # print(inputs.x.shape)
            with torch.no_grad():
                inputs.questions = model1.token_embedding(inputs.question).to(torch.float32).to('cuda')
            labels = batch.answer.to('cuda')
            # print(torch.max(batch.answer.reshape(-1,1000),dim=1))
            # labels = torch.argmax(batch.answer.reshape(-1,1000),dim=1).to('cuda')
            # print(labels)

            optimizer.zero_grad()

            # Forward pass
            outputs,l,l1,_ = model(inputs.x.float(),inputs.edge_index,inputs.questions,inputs.batch,inputs.question,inputs.global_feature)
            # labels = torch.argmax(labels.reshape(-1,1000),dim=1).to('cuda')
            # + loss_mse(l,l1)/outputs.size(0)
            # Compute the cross-entropy loss
            # + loss_mse(l,l1)/outputs.size(0)
            # + loss_mse(l,l1)/outputs.size(0)
            # + 0.1*loss_mse(l,l1)/outputs.size(0)
            loss = F.cross_entropy(outputs, labels) 

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            # Accumulate the total loss
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        scheduler.step()
        train_accuracy = correct_predictions / total_samples
        model.eval()
        val_loss=0.0
        correct_predictions=0
        total_samples=0
        for batch in val_dataloader:
            inputs= batch.to('cuda')
            if cfg.model.ncs:
                inputs.x = inputs.x[:,:512]
            with torch.no_grad():
                inputs.questions = model1.token_embedding(inputs.question).to(torch.float32).to('cuda')
            labels = batch.answer.to('cuda')
            # labels = (torch.max(batch.answer.reshape(-1,1000),dim=1)[0]).to('cuda')
            # labels = torch.argmax(batch.answer.reshape(-1,1000),dim=1).to('cuda')
            outputs,_,_,_ = model(inputs.x.float(),inputs.edge_index,inputs.questions,inputs.batch,inputs.question,inputs.global_feature)
            outputs = outputs.reshape(-1,1000)
            # labels = torch.argmax(labels.reshape(-1,1000),dim=1).to('cuda')
            loss = F.cross_entropy(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            # print(predicted)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        val_accuracy = correct_predictions / total_samples
        average_loss = total_loss / len(train_dataloader)
        average_val_loss = val_loss / len(val_dataloader)

        cross_loss=0.0
        correct_predictions=0
        total_samples=0
        for batch in cross_dataloader:
            inputs= batch.to('cuda')
            if cfg.model.ncs:
                inputs.x = inputs.x[:,:512]
            with torch.no_grad():
                inputs.questions = model1.token_embedding(inputs.question).to(torch.float32).to('cuda')
            # labels = (torch.max(batch.answer.reshape(-1,1000),dim=1)[0]).to('cuda')
            labels = batch.answer.to('cuda')
            # labels = torch.argmax(batch.answer.reshape(-1,1000),dim=1).to('cuda')
            outputs,_,_,_ = model(inputs.x.float(),inputs.edge_index,inputs.questions,inputs.batch,inputs.question,inputs.global_feature)
            outputs = outputs.reshape(-1,1000)
            # labels = torch.argmax(labels.reshape(-1,1000),dim=1).to('cuda')
            loss = F.cross_entropy(outputs, labels)
            cross_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            # print("---")
            # print(predicted)
            # print(labels)
            # print("--")
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        cross_accuracy = correct_predictions / total_samples
        average_cross_loss = val_loss / len(cross_dataloader)

        # if cross_best<=cross_accuracy:
        #     torch.save(model.state_dict(), './clip_gqa_v2_ours.pth')
        #     cross_best = cross_accuracy
        #     print("Model saved")

        # Print or log the training loss for the epoch
        print(f"Epoch {epoch + 1}/{cfg.data.num_epochs}, train Loss: {average_loss:.4f} train Accuracy: {train_accuracy:.4f}  val Loss: {average_val_loss:.4f} val Accuracy: {val_accuracy:.4f} cross Loss: {average_cross_loss:.4f} cross Accuracy: {cross_accuracy:.4f} ")
        # if val_accuracy >= best_accuracy:
        #     best_accuracy = max(best_accuracy,val_accuracy)
        # else:
        #     print("Stopping due to decrease in val accuracy")
        #     break
        # best_accuracy = max(best_accuracy,cross_accuracy)
    # return -best_accuracy
    # model.

if __name__ == "__main__":
    train()
