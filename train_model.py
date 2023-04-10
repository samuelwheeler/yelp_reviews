
import math
import os
import time
from tqdm import tqdm
import pandas as pd
from torch.optim.lr_scheduler import MultiStepLR
import text_classifier as TC
import yelp_dataset as YD
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np



# set hyperparameters and initial conditions
batch_size = 2048
depth = 4
dim = 128
sentence_length = 100
heads = 10
dropout = 0.1
embedding_dim = 50
num_classes = 5
epochs = 50
initial_lr = 0.0001
attention_type = 'quintic'



train_set = YD.YelpData(path = 'train_set')
test_set = YD.YelpData(path = 'test_set')


# device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
embedding_weights = torch.load('./embedding_weights.pt')
# define model:
model = TC.Review_Classifier(embedding_weights = embedding_weights, 
                             sentence_length = sentence_length,
                             embedding_dim = embedding_dim,
                             num_classes = num_classes,
                             dim = dim,
                             depth = depth,
                             heads = heads,
                             mlp_dim = dim,
                             dropout = dropout,
                             attention_type = attention_type)
starting_epoch = 0


model = nn.DataParallel(model)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr = initial_lr)






trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('1 star', '2 star', '3 star', '4 star', '5 star')



start_time = time.time()
#model = model.to(device)    
criterion = nn.CrossEntropyLoss()
#lambda1 = lambda epoch: 0.89**(2*epoch)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
train_accs = np.zeros(epochs)
test_accs = np.zeros(epochs)
learning_rates = np.zeros(epochs)

for epoch in range(epochs):
    
    lr = optimizer.param_groups[0]["lr"]
    print(f'Learning Rate: {lr}')
    learning_rates[epoch] = lr
    train_correct = 0
    train_total = 0    
    for batch_idx, (data, target) in enumerate(tqdm(trainloader)):
        if torch.cuda.is_available():
            data, target = data.to(device), target.to(device)
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        _, preds = torch.max(outputs.data, 1)
        train_correct += (preds == target).sum().item()
        train_total += target.size(0)
        if batch_idx%100 == 0:
            print(f'Loss: {loss.item()}')
    scheduler.step()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            model.eval()
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    train_acc, test_acc = train_correct/train_total, test_correct/test_total
    train_accs[epoch] = train_acc
    test_accs[epoch] = test_acc
    '''if epoch >= 2 and False:
        if test_accs[epoch] - test_accs[epoch-1] < 0.01:
            lr = lr * 0.75
            for g in optimizer.param_groups:
                g['lr'] = lr'''
    print(f'Epoch: {epoch + 1 + starting_epoch}, Train Acc: {train_acc}, Test Acc: {test_acc}')
total_time = time.time() - start_time


'''training_history = None
try:
    training_history = pd.read_csv('ViT_training_results')
except:
   training_history = None
   print('No training history found')'''



#df = pd.DataFrame({'train_accs':train_accs, 'test_accs':test_accs})

'''if training_history is not None:
    training_history = training_history.append(df).reset_index(drop = True)
    training_history.to_csv('ViT_training_results')
else:
    df.to_csv('ViT_training_results')'''

# df.to_csv('Vanilla_training_results_ablation_40.csv')

# state = {'epoch': starting_epoch + epochs, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
# torch.save(state, state_path)
