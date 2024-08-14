# from lavis.models import load_model_and_preprocess
from PIL import Image
import requests
import torch
import torch.nn as nn
import clip
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
from torch.optim import SGD,AdamW
from data.gen_dataloader import VQAv2Dataset
from tqdm import tqdm
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
class TransferModel(nn.Module):
    def __init__(self,num_classes=1000):
        super(TransferModel, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.classifier = nn.Linear(1024,num_classes)
    @torch.autocast(device_type="cuda")
    def forward(self,image,text):
        # sample = {"image": image, "text_input": text}
        # image = self.preprocess(image).unsqueeze(0).to(self.device)
        inputs = clip.tokenize(text).to(self.device)
        with torch.no_grad():
                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(inputs)

        # print(multimodal_emb.shape)
        multi_modal = torch.cat((image_features,text_features),dim=1)
        # print(multi_modal.dtype)
        # print(self.classifier.weight.dtype)
        out = self.classifier(multi_modal)
        return out
    
batch_size=128
num_workers=4
lr =1e-3
epochs = 50
momentum = 0.99
image_size = 224




mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

train_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomCrop(image_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=torch.tensor(mean),
        std=torch.tensor(std))
])

test_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

train_dataset = VQAv2Dataset('datasets/vqa_v2','train','VQAv2',transform=train_transform)
val_dataset = VQAv2Dataset('datasets/vqa_v2','val','VQAv2',transform=test_transform)
# cross_dataset = VQAv2Dataset('/raid/biplab/hassan/datasets/vqa_v2','val','VQAv2',transform=test_transform)

train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)
# cross_loader = DataLoader(cross_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)

transfer_model = TransferModel()

transfer_model = transfer_model.to('cuda')
optimizer = AdamW(transfer_model.parameters(), lr=lr)

optimizer.zero_grad()

train_loss_meter = AverageMeter()
val_loss_meter = AverageMeter()
# cross_loss_meter = AverageMeter()
train_accuracy_meter = AverageMeter()
val_accuracy_meter = AverageMeter()
# cross_accuracy_meter = AverageMeter()
best_val_acc = 0

for i in range(epochs):

    transfer_model.train()
    train_loss_meter.reset()
    train_accuracy_meter.reset()
    for data in tqdm(train_loader):
        img = data["img"]
        ques = data["question"]
        ans = data["answer"]
        img,ans = img.to('cuda'),ans.to('cuda')

        output = transfer_model(img,ques)
        # print(output.shape)
        # print(ans.shape)
        # print(ans)
        loss =  torch.nn.CrossEntropyLoss()(output,ans)
        train_loss_meter.update(loss.item(), img.size(0))
        # Calculate and update accuracy
        acc1 = accuracy(output, ans, topk=(1,))
        train_accuracy_meter.update(acc1[0].item(), img.size(0))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # break
    transfer_model.eval()
    val_loss_meter.reset()
    val_accuracy_meter.reset()
    for data in tqdm(val_loader):
        img = data["img"]
        ques = data["question"]
        ans = data["answer"]
        img,ans = img.to('cuda'),ans.to('cuda')

        output = transfer_model(img,ques)
        loss =  torch.nn.CrossEntropyLoss()(output,ans)
        val_loss_meter.update(loss.item(), img.size(0))
        # Calculate and update validation accuracy
        acc1 = accuracy(output, ans, topk=(1,))
        val_accuracy_meter.update(acc1[0].item(), img.size(0))
        # break
    # cross_loss_meter.reset()
    # cross_accuracy_meter.reset()
    # for data in tqdm(cross_loader):
    #     img = data["img"]
    #     ques = data["question"]
    #     ans = data["answer"]
    #     img,ans = img.to('cuda'),ans.to('cuda')

    #     output = transfer_model(img,ques)

    #     loss =  torch.nn.CrossEntropyLoss()(output,ans)
    #     # cross_loss_meter.update(loss.item(), img.size(0))
    #     # Calculate and update accuracy
    #     acc1 = accuracy(output, ans, topk=(1,))
        # cross_accuracy_meter.update(acc1[0].item(), img.size(0))
        # break
    # print(val_accuracy_meter.avg)
    # print(val_loss_meter.avg)
    print(f'Epoch: {i+1}, Validation Loss: {val_loss_meter.avg:.4f}, Validation Accuracy: {val_accuracy_meter.avg:.2f} ')
    if best_val_acc< val_accuracy_meter.avg:
        torch.save(transfer_model.state_dict(), './clip_vqa_v2.pth')
        best_val_acc = val_accuracy_meter.avg
        print("Model Saved!!!")

