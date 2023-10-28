import sys
sys.path.append()
from automation import Tester

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch

class SimpleNN(nn.Module):
    def __init__(self, input_size, h, output_size) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, h),
            nn.Tanh(),  
            nn.Linear(h, output_size),
        )

    def forward(self, x): return self.layers.forward(x)

def eval_model(test_loader, model):
    distances = []

    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            outputs = model(inputs)
            distances.append((outputs - labels)**2)

    return sum(distances)/len(distances)

def train_model(model, criterion, optimizer, train_loader, epochs):
    losses = []
    eval_loss = []
    model.train()
    for epoch in tqdm(range(epochs)):
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad() 
            outputs = model(inputs)
            outputs = torch.cat(list(outputs), dim=0)
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()
            # losses.append(abs(loss.item()))
        # eval_loss.append(eval_model(model))

    # plt.plot(range(len(losses)), losses, label = "loss")
    # plt.plot(range(0, len(eval_loss) * len(train_loader), len(train_loader)), eval_loss, label = "eval loss")
    # plt.show()

def test_nn_regression(x_train, x_test, y_train, y_test):

    input_size = 16
    output_size = 1
    learning_rate = 2e-4
    epochs = 500
    batch_size = 32
    h = 200

    model = SimpleNN(input_size, h, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    train_model(model, criterion, optimizer, train_loader, epochs)
    return eval_model(model)


path_to_data = "/Users/lucasvilsen/Desktop/DTU/MachineLearning&DataMining/Project2/StandardizedDataFrameWithNansFilled.csv"
tester = Tester("LifeExpectancyRegression", path_to_data, function_to_test = test_nn_regression, final_test = False, k = 10)

