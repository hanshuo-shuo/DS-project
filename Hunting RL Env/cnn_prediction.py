import numpy as np
from gif_to_array import readGif
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
CHECKPOINT_PATH = './chase.tar'
import wandb

PROJECT_NAME = 'chase_and_escape'

run = wandb.init(project=PROJECT_NAME, resume=False)

def sample_d(num):
    gif_array = readGif(f"gif_chase/test_{num}.gif")
    t_start = np.random.randint(0,len(gif_array))
    mid_d = min(10, len(gif_array) - t_start)
    t_end = np.random.randint(t_start, mid_d+t_start)

    image_start = gif_array[t_start]
    image_end = gif_array[t_end]
    distance = t_end - t_start
    if distance == 0:
        bar = [1, 0, 0, 0]

    elif distance in [1, 2, 3]:
        bar = [0, 1, 0, 0]

    elif distance in [4, 5, 6]:
        bar = [0, 0, 1, 0]

    else:
        bar = [0, 0, 0, 1]

    return image_start, image_end, bar


def train_set(n):
    train_set = []

    for i in range(n):
        num = np.random.randint(0,3000)
        image_start, image_end, distance = sample_d(num)
        sample = np.concatenate((image_start, image_end), axis=1)
        train_set.append([sample, distance])
    return train_set

def accuracy(output, target, topk=(2,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    res = res[0].numpy()
    return res


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(119040, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 4)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc3(x)
        # output = F.log_softmax(x, dim=1)
        return x


def train_and_val(n):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a = train_set(n)
    train_x = []
    label_y = []
    test_size = 0.5
    for i in range(n):
        img = a[i][0]
        train_x.append(img)
        label = a[i][1]
        label_y.append(label)

    train_x = np.array(train_x)
    train_x = np.float32(train_x)
    label_y = np.array(label_y)
    label_y = np.float32(label_y)
    train_x, val_x, train_y, val_y = train_test_split(train_x, label_y, test_size=test_size)

    train_x = train_x.reshape(int(n * test_size), 1, 64, 128)
    train_x = torch.from_numpy(train_x)

    # train_y = train_y.astype(int)
    train_y = torch.from_numpy(train_y)
    val_x = val_x.reshape(int((1-test_size) * n), 1, 64, 128)
    val_x = torch.from_numpy(val_x)

    # converting the target into torch format
    # val_y = val_y.astype(int)
    val_y = torch.from_numpy(val_y)

    x_train, y_train = Variable(train_x), Variable(train_y)
    x_val, y_val = Variable(val_x), Variable(val_y)

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    return x_train, y_train, x_val, y_val

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCH = 20000
    BATCH_SIZE = 10
    LR = 0.0001
    cnn = CNN().to(device)

    # optimizer = torch.optim.SGD(cnn.parameters(), lr=LR, weight_decay=0.001)  # optimize all cnn parameters
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=0.004)  # optimize all cnn parameters
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()


    # clearing the Gradients of the model parameters
    optimizer.zero_grad()
    # empty list to store training losses
    train_losses = []
    # empty list to store validation losses
    val_losses = []
    p_traines = []
    p_vales = []


    # training and testing
    for epoch in range(EPOCH):
        x_train, y_train, x_val, y_val = train_and_val(BATCH_SIZE)
        output_train = cnn(x_train)
        # output_val = cnn(x_val)
        loss_train = criterion(output_train, y_train)
        # loss_val = criterion(output_val, y_val)


        # computing the updated weights of all the model parameters
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if epoch % 20 == 0:
            #print(y_train)
            #print(output_train)
            print('Epoch : ', epoch + 1, '\t', 'train_loss :', loss_train)
            #print('Epoch : ', epoch + 1, '\t', 'val_loss :', loss_val)
            train_losses.append(loss_train)
            #val_losses.append(loss_val)
            wandb.log({'loss_train': loss_train})
            with torch.no_grad():
                output = cnn(x_train)
                output = F.softmax(output, dim=1)
                prob_t = list(output.numpy())
                predictions = np.argmax(prob_t, axis=1)
                real = np.argmax(y_train, axis=1)
                # accuracy on training set
                ## top2
                p_train = accuracy(output, real)
                ## top1
                pt = accuracy_score(real, predictions)
            with torch.no_grad():
                output = cnn(x_val)
                output = F.softmax(output, dim=1)
                # output = torch.exp(output)
                prob_v = list(output.numpy())
                predictions = np.argmax(prob_v, axis=1)
                real = np.argmax(y_val, axis=1)
                ## top 1
                pv = accuracy_score(real, predictions)
                ## top 2
                p_val = accuracy(output, real)

            # p_traines.append(p_train)
            # p_vales.append(p_val)
            print("prob on the training set is:", prob_t, "prob on the val set is:", prob_v)

            wandb.log({'Top2_acc_train': p_train, 'Top2_acc_val': p_val})
            wandb.log({'Top1_acc_train': pt, 'Top1_acc_val': pv})

            #Save our checkpoint loc
            torch.save({
                'model_state_dict': cnn.state_dict()
            }, CHECKPOINT_PATH)



def main1():
    x_train, y_train, x_val, y_val = train_and_val(20)
    print(x_val)
    print(y_val)


if __name__ == "__main__":
    main()