import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
import requests
import pandas as pd
import datetime
import math
import config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_size = 100
days = 100

symbol = "GOOGL"

api_key = config.API_KEY
secret_key = config.SECRET_KEY

headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}


def convert_bars(bars, train_test, linear):
    line = ""
    if linear == False:
        open = bars[-1]["o"]
        close = bars[-1]["c"]
        if close < open:
            line += "0,1,0,"
        elif close > open:
            line += "1,0,0,"
        elif close == open:
            line += "0,0,1,"
        bars = bars[:-1]
        for bar in bars:
            line += str(bar["c"]) + ","
        return line
    elif linear == True:
        line += str(bars[-1]["c"]) + ","
        bars = bars[:-1]
        for bar in bars:
            line += str(bar["c"]) + ","
        return line
    elif train_test == False:
        for bar in bars:
            line += str(bar["c"]) + ","
        return line


def get_bars(
    symbol, timeframe, headers, train, limit=2000, sample_size=100, train_percentage=0.9
):
    if train == True:
        bars = []
        for i in range(days):
            end_date = datetime.datetime.today() - datetime.timedelta(days=i + days + 1)
            end_date = end_date.replace(hour=21, minute=0, second=0, microsecond=0)
            s_end_date = str(end_date)
            year = s_end_date[:4]
            month = s_end_date[5:7]
            day = s_end_date[8:10]
            hour = s_end_date[11:13]
            end = year + "-" + month + "-" + day + "T" + hour + ":00:00Z"

            start_date = end_date - datetime.timedelta(days=1)
            s_start_date = str(start_date)
            year = s_start_date[:4]
            month = s_start_date[5:7]
            day = s_start_date[8:10]
            hour = s_start_date[11:13]
            start = year + "-" + month + "-" + day + "T" + hour + ":30:00Z"

            r = requests.get(
                "https://data.alpaca.markets/v2/stocks/{}/bars?timeframe={}&start={}&end={}&limit={}".format(
                    symbol, timeframe, start, end, limit
                ),
                headers=headers,
            )
            day_bars = r.json()
            day_bars = day_bars["bars"]
            if day_bars:
                bars += day_bars
        filename = "Pytorch/data/bars/{}_training_bars.txt".format(symbol)
        f = open(filename, "w+")
        for i in range((len(bars)) - sample_size):
            if i == ((len(bars)) - sample_size) - 1:
                new_bars = bars[i : i + int((sample_size + 1))]
                converted_bars = convert_bars(new_bars, train_test=True, linear=False)
                f.write(converted_bars)
            else:
                new_bars = bars[i : i + int((sample_size + 1))]
                converted_bars = convert_bars(new_bars, train_test=True, linear=False)
                f.write(converted_bars + "\n")
    elif train == False:
        bars = []
        for i in range(days):
            end_date = datetime.datetime.today() - datetime.timedelta(days=i + 1)
            end_date = end_date.replace(hour=21, minute=0, second=0, microsecond=0)
            s_end_date = str(end_date)
            year = s_end_date[:4]
            month = s_end_date[5:7]
            day = s_end_date[8:10]
            hour = s_end_date[11:13]
            end = year + "-" + month + "-" + day + "T" + hour + ":00:00Z"

            start_date = end_date - datetime.timedelta(days=1)
            s_start_date = str(start_date)
            year = s_start_date[:4]
            month = s_start_date[5:7]
            day = s_start_date[8:10]
            hour = s_start_date[11:13]
            start = year + "-" + month + "-" + day + "T" + hour + ":30:00Z"

            r = requests.get(
                "https://data.alpaca.markets/v2/stocks/{}/bars?timeframe={}&start={}&end={}&limit={}".format(
                    symbol, timeframe, start, end, limit
                ),
                headers=headers,
            )
            day_bars = r.json()
            day_bars = day_bars["bars"]
            if day_bars:
                bars += day_bars
        filename = "Pytorch/data/bars/{}_test_bars.txt".format(symbol)
        f = open(filename, "w+")
        for i in range((len(bars)) - sample_size):
            if i == ((len(bars)) - sample_size) - 1:
                new_bars = bars[i : i + int((sample_size + 1))]
                converted_bars = convert_bars(new_bars, train_test=True, linear=False)
                f.write(converted_bars)
            else:
                new_bars = bars[i : i + int((sample_size + 1))]
                converted_bars = convert_bars(new_bars, train_test=True, linear=False)
                f.write(converted_bars + "\n")


get_bars(symbol, "1Day", headers, train=True)
get_bars(symbol, "1Day", headers, train=False)


class BarsDataset(Dataset):

    def __init__(self, file, transform=None):
        list = []
        for i in range(sample_size + 3):
            list.append(i)
        xy = np.loadtxt(file, delimiter=",", dtype=np.float32, usecols=list)
        self.n_samples = xy.shape[0]

        sc = StandardScaler()
        self.x_data = xy[:, 3:]
        self.x_data = sc.fit_transform(self.x_data)
        self.y_data = xy[:, [0, 1, 2]]

        self.n_features = self.x_data.shape[-1]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        inputs, labels = sample
        return torch.from_numpy(inputs), torch.from_numpy(labels)


class Model(nn.Module):
    def __init__(self, n_input_features, hidden_layer_size):
        super(Model, self).__init__()
        self.n_input_features = n_input_features
        self.linear1 = nn.Linear(n_input_features, 3)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.initialize_weights()

    def forward(self, x):
        out = self.linear1(x)
        print(out)
        out = self.sigmoid(out)
        print(out)
        print("#####################")

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)


def get_class_weights(file):
    labels = np.loadtxt(file, delimiter=",", dtype=np.float32, usecols=[0, 1, 2])
    class_0 = 0
    class_2 = 0
    class_1 = 0
    for label in labels:
        if label[0] == 1.0:
            class_0 += 1
        elif label[1] == 1.0:
            class_1 += 1
        else:
            class_2 += 1
    classes = [class_0, class_1, class_2]
    class_weights = []
    for class_ in classes:
        class_weight = max(classes) / class_
        class_weights.append(class_weight)
    print(class_weights)
    return class_weights


batch_size = 128
composed = transforms.Compose([ToTensor()])
train_dataset = BarsDataset(
    "Pytorch tests/data/bars/{}_training_bars.txt".format(symbol), transform=composed
)


def get_loader(dataset, batch_size, class_weights):
    sample_weights = [0] * len(dataset)

    for i, (data, label) in enumerate(dataset):
        if label[0] == 1.0:
            class_index = 0
        elif label[1] == 1.0:
            class_index = 1
        else:
            class_index = 2
        class_index = int(class_index)
        class_weight = class_weights[class_index]
        sample_weights[i] = class_weight

    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
    )
    return loader


train_loader = get_loader(
    train_dataset,
    batch_size=batch_size,
    class_weights=get_class_weights(
        "Pytorch tests/data/bars/{}_training_bars.txt".format(symbol)
    ),
)
test_dataset = BarsDataset(
    "Pytorch tests/data/bars/{}_test_bars.txt".format(symbol), transform=composed
)
test_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

n_features = train_dataset.n_features
learning_rate = 0.1
momentum = 0.9
hidden_layer_size = 50
model = Model(n_features, hidden_layer_size)
# model.load_state_dict(torch.load('Pytorch test/models/Logistic regression'))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min")
num_epochs = 100


def train(train_loader):
    model.train()
    total_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss
        for param in model.sigmoid.parameters():
            param.requires_grad = False
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss


def train_1(train_loader):
    total_loss = 0
    model.train()
    inputs, labels = next(iter(train_loader))
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    total_loss += loss
    for param in model.sigmoid.parameters():
        param.requires_grad = False
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # print(model.linear4.weight.grad)
    optimizer.step()
    # print(model.linear4.weight.grad)
    optimizer.zero_grad()
    return total_loss


def test(test_loader):
    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for inputs, labels in test_loader:
            outputs = model(inputs)
            n_samples += labels.size(0)
            n_correct += (outputs == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f"Accuracy of the network on test data: {acc} %")


def predict(input):
    with torch.no_grad():
        input = input.split(",")
        output = model(input)
        return output


if __name__ == "__main__":
    for epoch in range(num_epochs):
        val_loss = train_1(train_loader)
        scheduler.step(val_loss)
        if (epoch + 1) % 10 == 0:
            print(f"epoch: {epoch+1}, loss = {val_loss:.4f}")
    for epoch in range(num_epochs):
        val_loss = train(train_loader)
        scheduler.step(val_loss)
        if (epoch + 1) % 10 == 0:
            print(
                f"epoch: {epoch+1}, loss = {val_loss/(train_dataset.n_samples/batch_size):.4f}"
            )
    test(test_loader)
    torch.save(model.state_dict(), "Pytorch test/models/Logistic_regression")
