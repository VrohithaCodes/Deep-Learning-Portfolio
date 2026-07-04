import torch
import torch.nn as nn
import torch.optim as optim
class ANN(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(784,128)
        self.layer2 = nn.Linear(128,10)


    def forward(self,x):

        x = self.layer1(x)

        x = torch.relu(x)

        x = self.layer2(x)

        return x
model = ANN()

print(model)
input_data = torch.randn(1,784)

output = model(input_data)

print(output)    
for name, param in model.named_parameters():

    print(name,param.shape)
