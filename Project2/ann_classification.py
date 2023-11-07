import sys
sys.path.append("/Users/lucasvilsen/Desktop/DTU/MachineLearning&DataMining/Project2")
from automation import Tester

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

class SimpleNN(nn.Module):
    def __init__(self, input_size, h, output_size) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, h),
            nn.ReLU(),  
            nn.Linear(h, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x): return self.layers.forward(x)

# def eval_model(test_loader, model):
#     model.eval()
#     total_distance = 0
#     num_samples = 0

#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             outputs = model(inputs).squeeze()
#             batch_distance = (torch.square(outputs - labels)).sum()
#             total_distance += batch_distance 
#             num_samples += len(labels)

#     return total_distance / num_samples

def get_predictions(test_x, model):
    model.eval()
    with torch.no_grad():
        return model(test_x)

def train_model(model, criterion, optimizer, train_loader, epochs):
    losses = []
    model.train()
    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad() 
            outputs = model(inputs)
            outputs = torch.cat(list(outputs), dim=0)
            labels = torch.flatten(labels)
            loss = criterion(outputs, labels.float()) 
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    # plt.plot(losses)
    # plt.show()


def ann(x_train, x_test, y_train, y_test, func_var):

    input_size = 16
    # binary for status
    output_size = 2
    learning_rate = 1e-5
    epochs = 50
    batch_size = 32
    h = func_var

    model = SimpleNN(input_size, h, output_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_model(model, criterion, optimizer, train_loader, epochs)
    raw_predictions = get_predictions(x_test, model)
    predictions = np.argmax(raw_predictions, axis=1)
    y_test = np.argmax(y_test, axis=1)
    print("done")

    return f1_score(predictions, y_test)

if __name__ == "__main__":
    path_to_data = "/Users/lucasvilsen/Desktop/DTU/MachineLearning&DataMining/Project2/StandardizedDataFrameWithNansFilled.csv"
    # h_to_test = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    h_to_test = 2048
    # print(h_to_test)
    tester = Tester("StatusClassification", path_to_data, function_to_test = ann, final_test = True, k = 10, vars_to_test=h_to_test)