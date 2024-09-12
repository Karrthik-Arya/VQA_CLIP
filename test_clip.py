import torch
import torch.nn as nn
from data.testDataset import TestDataset
from torch.utils.data import DataLoader
import clip
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

transfer_model = TransferModel()
transfer_model = transfer_model.to('cuda')
transfer_model.load_state_dict(torch.load('./clip_vqa_v2.pth'))
transfer_model.eval()   

test_dataset = TestDataset('datasets/test/images', 'datasets/test/test_questions.csv')

batch_size=128
num_workers=4

test_loader = DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)

test_accuracy_meter = AverageMeter()

for data in tqdm(test_loader):
    img = data["img"]
    ques = data["question"]
    ans = data["answer"]
    img,ans = img.to('cuda'),ans.to('cuda')

    output = transfer_model(img,ques)
    loss =  torch.nn.CrossEntropyLoss()(output,ans)
    # val_loss_meter.update(loss.item(), img.size(0))
    # Calculate and update validation accuracy
    acc1 = accuracy(output, ans, topk=(1,))
    test_accuracy_meter.update(acc1[0].item(), img.size(0))

print(f'Test Accuracy: {test_accuracy_meter.avg:.2f} ')
