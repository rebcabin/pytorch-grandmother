import torch


class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        in_features = 100
        mid_features = 200
        out_features = 10

        self.linear1 = torch.nn.Linear(in_features, mid_features)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(mid_features, out_features)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


