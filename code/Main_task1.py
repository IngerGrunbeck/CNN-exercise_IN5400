

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from torch import Tensor

import time
import os
import numpy as np

import PIL.Image
from sklearn.metrics import average_precision_score as AP

from vocparseclslabels import PascalVOC
from GUI import run_GUI

from typing import Callable, Optional


class dataset_voc(Dataset):
    def __init__(self, root_dir, trvaltest, transform=None):
        self.root_dir = root_dir
        self.trvaltest = trvaltest
        self.transform = transform
        self.imgfilenames = []
        self.label = []

        pv = PascalVOC(self.root_dir)

        if trvaltest == 0:
            fn = os.path.join(pv.set_dir, 'train.txt')
            with open(fn, 'r') as f:
                for line in f:
                    v = line.rstrip()
                    nm = os.path.join(pv.img_dir, v+'.jpg')
                    self.imgfilenames.append(nm)
                    #print(nm)
                    self.label.append(torch.as_tensor([0]*len(pv.list_image_sets())))

            for i, cat in enumerate(pv.list_image_sets()):
                ls = pv.imgs_from_category_as_list(cat, 'train')
                for image_name in ls:
                    img_id = self.imgfilenames.index(os.path.join(pv.img_dir, image_name +'.jpg'))
                    self.label[img_id][i] = 1

        if trvaltest == 1:
            fn = os.path.join(pv.set_dir, 'val.txt')
            with open(fn, 'r') as f:
                for line in f:
                    v = line.rstrip()
                    nm = os.path.join(pv.img_dir, v+'.jpg')
                    self.imgfilenames.append(nm)
                    #print(nm)
                    self.label.append(torch.as_tensor([0]*len(pv.list_image_sets())))

            for i, cat in enumerate(pv.list_image_sets()):
                ls = pv.imgs_from_category_as_list(cat, 'val')
                for image_name in ls:
                    img_id = self.imgfilenames.index(os.path.join(pv.img_dir, image_name +'.jpg'))
                    self.label[img_id][i] = 1


    def __len__(self):
        return len(self.imgfilenames)

    def __getitem__(self, idx):

        image = PIL.Image.open(self.imgfilenames[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.label[idx]

        sample = {'image': image, 'label': label, 'filename': self.imgfilenames[idx]}

        return sample


def train_epoch(model,  trainloader,  criterion, device, optimizer):

    model.train()
 
    losses = []
    for batch_idx, data in enumerate(trainloader):

        inputs = data['image'].to(device)
        labels = data['label'].to(device)
        labels = labels.float()

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if batch_idx % 100 == 0:
            print('current mean of losses ', np.mean(losses))
    return np.mean(losses)


def evaluate_meanavgprecision(model, dataloader, criterion, device, numcl):

    model.eval()
    
    concat_pred=[np.empty(shape=(0)) for _ in range(numcl)] #prediction scores for each class. each numpy array is a list of scores. one score per image
    concat_labels=[np.empty(shape=(0)) for _ in range(numcl)] #labels scores for each class. each numpy array is a list of labels. one label per image
    avgprecs=np.zeros(numcl) #average precision for each class
    fnames = [] #filenames as they come out of the dataloader
    classifier = torch.nn.Sigmoid()
    
    
    with torch.no_grad():
        losses = []
        for batch_idx, data in enumerate(dataloader):
      
      
            if (batch_idx%100==0) and (batch_idx>=100):
                print('at val batchindex: ', batch_idx)
      
            inputs = data['image'].to(device)
            outputs = model(inputs)
            cpu_outputs = outputs.to('cpu')

            labels = data['label']
            labels = labels.float()

            loss = criterion(cpu_outputs, labels)
            losses.append(loss.item())

            sigmoid_output = classifier(cpu_outputs)
            for img in sigmoid_output:
                for cat_idx in range(numcl):
                    concat_pred[cat_idx] = np.append(concat_pred[cat_idx], img[cat_idx])
            for img in labels:
                for cat_idx in range(numcl):
                    concat_labels[cat_idx] = np.append(concat_labels[cat_idx], img[cat_idx])
            for name in data['filename']:
                fnames.append(name)

    for c in range(numcl):   
        avgprecs[c] = AP(y_true=concat_labels[c], y_score=concat_pred[c])

      
    return avgprecs, np.mean(losses), concat_labels, concat_pred, fnames


def traineval2_model_nocv(dataloader_train, dataloader_test,  model,  criterion, optimizer, scheduler, num_epochs, device, numcl):

    best_measure = 0
    best_epoch = -1

    trainlosses=[]
    testlosses=[]
    testperfs=[]
  
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        avgloss=train_epoch(model,  dataloader_train,  criterion,  device , optimizer)
        trainlosses.append(avgloss)

        if scheduler is not None:
            scheduler.step()

        perfmeasure, testloss, concat_labels, concat_pred, fnames = evaluate_meanavgprecision(model, dataloader_test, criterion, device, numcl)
        testlosses.append(testloss)
        testperfs.append(perfmeasure)

        print('at epoch: ', epoch, ' classwise perfmeasure ', perfmeasure)

        avgperfmeasure = np.mean(perfmeasure)
        print('at epoch: ', epoch, ' avgperfmeasure ', avgperfmeasure)

        if avgperfmeasure > best_measure:
            bestweights= model.state_dict()
            best_measure = avgperfmeasure
            best_epoch = epoch
            print('current best', best_measure, ' at epoch ', best_epoch)
            best_concat_pred = concat_pred
            best_concat_labels = concat_labels
            best_fname = fnames

    save_scores(best_concat_pred, best_fname)
    tailacc(image_scores=best_concat_pred, labels=best_concat_labels, cat_nr=numcl)


    return best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs


class BCEWithLogitsLoss(nn.modules.loss._Loss):

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean',
                 pos_weight: Optional[Tensor] = None) -> None:
        super(BCEWithLogitsLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        assert self.pos_weight is None or isinstance(self.pos_weight, Tensor)
        return torch.nn.functional.binary_cross_entropy_with_logits(input, target,
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction=self.reduction)


def save_scores(scores, filenames):

    #pv = PascalVOC(root_dir = '../data/VOCdevkit/VOC2012/')
    pv = PascalVOC(root_dir='/itf-fi-ml/shared/IN5400/dataforall/mandatory1/VOCdevkit/VOC2012/')


    for i, class_name in enumerate(pv.list_image_sets()):
        temp_list = zip(scores[i].tolist(), filenames)
        sorted_pairs = sorted(temp_list, reverse=True)
        tuples = zip(*sorted_pairs)
        list1, list2 = [list(tuple) for tuple in tuples]
        sorted_scores, fnames = np.array(list1), np.array(list2)

        np.save('../saved_scores/' + class_name + '_scores', sorted_scores)
        np.save('../saved_scores/' + class_name + '_filenames', fnames)


def test_train_curve(epochs, labels='loss', train_data=None, test_data=None):
    plt.figure()
    plt.xticks(np.arange(0, epochs, step=2))
    plt.title('Learning Curve')

    if labels == 'loss':
        if train_data:
            plt.plot(train_data, label='training')
        if test_data:
            plt.plot(test_data, label='validation')
        plt.ylabel('Loss')
        plt.legend()
    elif labels == 'map':
        avg = []
        for element in test_data:
            avg.append(np.mean(element))
        plt.plot(avg)
        plt.ylabel('mAP')

    plt.xlabel('Epochs')
    plt.show()

def tailacc(image_scores, labels, cat_nr):
    thresshold = 0.5
    t_range = 15
    avg_tailacc = []
    tailacc_t = [np.empty(shape=(0)) for _ in range(t_range)]


    for cat in range(cat_nr):
        t_max = max(image_scores[cat])
        stepsize = (t_max - thresshold)/(t_range)
        t_list = [0.5]
        for _ in range(t_range - 1):
            t_list.append(t_list[-1] + stepsize)


        binary_scores = [int(x > thresshold) for x in image_scores[cat].tolist()]
        hard_classifier = [1 if i == j else 0 for i, j in zip(labels[cat].tolist(), binary_scores)]
        for i, t in enumerate(t_list):
            soft_classifier = [int(x > t) for x in image_scores[cat].tolist()]
            tailacc_t[i] = np.append(tailacc_t[i], (1/sum(soft_classifier)*sum([a * b for a, b in zip(hard_classifier, soft_classifier)])))

    for t_avg in tailacc_t:
        avg_tailacc.append(np.mean(t_avg))

    plt.figure()
    plt.plot([round(t, 2) for t in t_list], avg_tailacc)
    plt.title('Average Tailaccuracy')
    plt.xlabel('t')
    plt.show()

def runstuff():
    np.random.seed(9400)
    torch.manual_seed(9400)

    config = dict()
  
    config['use_gpu'] = True
    config['lr']=0.005
    config['batchsize_train'] = 16
    config['batchsize_val'] = 64
    config['maxnumepochs'] = 35
    config['scheduler_stepsize'] = 10
    config['scheduler_factor'] = 0.3

    # kind of a dataset property
    config['numcl']=20

    # data augmentations
    data_transforms = {
        'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    #datasets
    image_datasets={}
    #image_datasets['train'] = dataset_voc(root_dir='../data/VOCdevkit/VOC2012/',trvaltest=0, transform=data_transforms['train'])
    #image_datasets['val'] = dataset_voc(root_dir='../data/VOCdevkit/VOC2012/',trvaltest=1, transform=data_transforms['val'])
    image_datasets['train']=dataset_voc(root_dir='/itf-fi-ml/shared/IN5400/dataforall/mandatory1/VOCdevkit/VOC2012/',trvaltest=0, transform=data_transforms['train'])
    image_datasets['val']=dataset_voc(root_dir='/itf-fi-ml/shared/IN5400/dataforall/mandatory1/VOCdevkit/VOC2012/',trvaltest=1, transform=data_transforms['val'])

    #dataloaders
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=config['batchsize_train'], num_workers=1, shuffle=True)
    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=config['batchsize_val'], num_workers=1, shuffle=False)

    # device
    if config['use_gpu']:
        device = torch.device('cuda:0')

    else:
        device = torch.device('cpu')

    # model
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config['numcl'])
    model.fc.reset_parameters()
  
    model = model.to(device)

    lossfct = BCEWithLogitsLoss()

    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)

    # Decay LR by a factor of 0.3 every X epochs
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=config['scheduler_stepsize'], gamma=config['scheduler_factor'], last_epoch=-1)

    best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs = traineval2_model_nocv(dataloaders['train'], dataloaders['val'] ,  model ,  lossfct, optimizer, scheduler, num_epochs= config['maxnumepochs'], device = device , numcl = config['numcl'])
    print('Best', best_measure, ' at epoch ', best_epoch)

    # saving the model
    torch.save(bestweights, '../saved_model/best_model_state.pth')
    model.load_state_dict(bestweights)
    torch.save(model, '../saved_model/best_entire_model.pth')

    # plotting the learning curves
    test_train_curve(epochs=config['maxnumepochs'], labels='loss', train_data=trainlosses, test_data=testlosses)
    test_train_curve(epochs=config['maxnumepochs'], labels='map', test_data=testperfs)


if __name__ == '__main__':
    runstuff()
    gui = run_GUI(root_dir='../saved_scores/', classes=[
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor'], img_nr=10)

    gui.main()
