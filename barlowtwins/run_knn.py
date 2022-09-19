import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
import torchvision
import pytorch_lightning as pl
import lightly
import tqdm
import time

train_set = datasets.CIFAR10(root='./data', train=True, download=True)
val_set = datasets.CIFAR10(root='./data', train=False)

NUM_SAMPLES = 1000

train_indices = random.sample(range(len(train_set)), k=NUM_SAMPLES)
train_subset = train_set.data[train_indices]
train_subset_labels = np.array(train_set.targets)[train_indices]

val_indices = random.sample(range(len(val_set)), k=NUM_SAMPLES)
val_subset = val_set.data[val_indices]
val_subset_labels = np.array(val_set.targets)[val_indices]


def predict_knn(sample, train_data, train_labels, k):
    '''
    returns the predicted label for a specific validation sample
    
    :param sample: single example from validation set
    :param train_data: full training set as a single array
    :param train_labels: full set of training labels and a single array
    :param k: number of nearest neighbors used for k-NN voting
    '''
    data = train_data.reshape(NUM_SAMPLES, -1)
    label_count = np.zeros(10)
    dist = np.sum(np.abs(sample.flatten() - data),axis=1)
    idx = np.argpartition(dist,k)
    min_ind = idx[:k]
    for x in min_ind:
        label_count[train_labels[x]] +=1
    return np.argmax(label_count)


predictions_7 = []
predictions_13 = []
predictions_19 = []

start = time.time()
for sample in tqdm.tqdm(val_subset):
    predictions_7.append(predict_knn(sample, train_subset, train_subset_labels, k=7))
    predictions_13.append(predict_knn(sample, train_subset, train_subset_labels, k=13))
    predictions_19.append(predict_knn(sample, train_subset, train_subset_labels, k=19))
end=time.time()

print('Time elapsed: ', (start-end)/60)


matches_7 = (np.array(predictions_7) == val_subset_labels)
accuracy_7 = np.sum(matches_7)/NUM_SAMPLES * 100
print(f"k-NN accuracy (k=7): {accuracy_7}%")

matches_13 = (np.array(predictions_13) == val_subset_labels)
accuracy_13 = np.sum(matches_13)/NUM_SAMPLES * 100
print(f"k-NN accuracy (k=13): {accuracy_13}%")

matches_19 = (np.array(predictions_19) == val_subset_labels)
accuracy_19 = np.sum(matches_19)/NUM_SAMPLES * 100
print(f"k-NN accuracy (k=19): {accuracy_19}%")