# USAGE
# python train_covid19.py --dataset generated_dataset

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import time
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
from imutils import paths
from tqdm import tqdm
from vit_pytorch.distill import DistillableViT, DistillWrapper
import numpy as np
import argparse
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default='generated_dataset',
                help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="covid19.model",
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 100
BS = 128

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths_Positive = list(paths.list_images(r'generated_dataset\covid'))
imagePaths_Negative = list(paths.list_images(r'generated_dataset\normal'))
data = []
labels = []

label_dict = {'normal': 0, 'covid': 1}

# loop over the image paths
for imagePath in tqdm(imagePaths_Positive, desc='loading positives'):
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    # load the image, swap color channels, and resize it to be a fixed
    # 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))

    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label_dict[label])

# loop over the image paths
for imagePath in tqdm(imagePaths_Negative, desc='loading negatives'):
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    # load the image, swap color channels, and resize it to be a fixed
    # 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))

    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label_dict[label])

# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 255]
data = np.array(data) / 255.0
# data = np.array(data)
labels = np.array(labels)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
print("[INFO] Splitting train and test dataset...")
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20, stratify=labels, random_state=42)


class Covid19Dataset(Dataset):
    def __init__(self, dataX, dataY):
        super(Covid19Dataset, self).__init__()
        self.dataX = dataX
        self.dataY = dataY

    def __len__(self):
        return len(self.dataX)

    def __getitem__(self, item):
        return self.dataX[item], self.dataY[item]


print("[INFO] Deployment for distilling VI transformer...")

train_dataset = Covid19Dataset(dataX=trainX, dataY=trainY)
test_dataset = Covid19Dataset(dataX=testX, dataY=testY)

train_dataLoader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
test_dataLoader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)

teacher = resnet50(pretrained=True)

v = DistillableViT(
    image_size=128,
    patch_size=16,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=8,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

distiller = DistillWrapper(
    student=v,
    teacher=teacher,
    temperature=3,  # temperature of distillation
    alpha=0.5,  # trade between main loss and distillation loss
    hard=False  # whether to use soft or hard distillation
).cuda()
learning_rate = 5e-5
optimizer = torch.optim.AdamW(lr=learning_rate, params=distiller.parameters())
print("[INFO] Distilling VI transformer for covid-19...")
epochs = 4
for epoch in range(epochs):
    all_loss = []
    time.sleep(0.1)
    for (img_, labels_) in tqdm(train_dataLoader, desc='distill epoch: {}'.format(epoch + 1)):
        img = img_.permute(0, 3, 1, 2).float().cuda()
        labels = labels_.squeeze(dim=-1).long().cuda()
        distiller.zero_grad()
        loss = distiller(img, labels)
        loss.backward()
        optimizer.step()
        all_loss.append(loss.item())
    time.sleep(0.1)
    print('Distilling epoch: {}, loss: {}'.format(epoch, np.mean(all_loss)))
    time.sleep(0.1)
    gold_labels, pred_labels = [], []
    for (img_, labels_) in tqdm(test_dataLoader, desc='test epoch: {}'.format(epoch + 1)):
        img = img_.permute(0, 3, 1, 2).float().cuda()
        labels = labels_.squeeze(dim=-1).numpy().tolist()
        batch_preds = v(img)
        batch_logits = torch.softmax(batch_preds[:, :2], dim=-1)
        batch_preds_idx = torch.argmax(batch_logits, dim=-1).cpu().detach().numpy().tolist()
        pred_labels.extend(batch_preds_idx)
        gold_labels.extend(labels)
    acc = accuracy_score(y_true=np.array(gold_labels), y_pred=np.array(pred_labels))
    maP, maR, maF, _ = precision_recall_fscore_support(y_true=np.array(gold_labels), y_pred=np.array(pred_labels),
                                                       average='macro')
    miP, miR, miF, _ = precision_recall_fscore_support(y_true=np.array(gold_labels), y_pred=np.array(pred_labels),
                                                       average='micro')

    print('Distill Results in Epoch: {}'.format(epoch))
    print('total accuracy: {:.6%}')
    print('Macro Metric'.center(20,'='))
    print('precision: {:.6%}'.format(maP))
    print('recall: {:.6%}'.format(maR))
    print('f1-score: {:.6%}'.format(maF))

    print('Micro Metric'.center(20, '='))
    print('precision: {:.6%}'.format(miP))
    print('recall: {:.6%}'.format(miR))
    print('f1-score: {:.6%}'.format(miF))
    time.sleep(0.1)