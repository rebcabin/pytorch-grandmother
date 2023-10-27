import torch
import torch.nn as nn
import torch.optim as optim


EPOCHS = 13
BATCH_SIZE = 5
IN_FEATURES = 3
OUT_FEATURES = 2
BMM_FEATURES = 25  # should be B^2

EPOCH_DIM = 0
BATCH_DIM = 1
OUT_FEATURE_DIM = 2


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a1 = nn.Linear(IN_FEATURES, IN_FEATURES)
        self.a2 = nn.Linear(IN_FEATURES, IN_FEATURES)
        self.a3 = nn.Linear(BMM_FEATURES, OUT_FEATURES)

    def forward(self, x_):
        o1 = self.a1(x_)
        o2 = self.a2(x_).transpose(BATCH_DIM, OUT_FEATURE_DIM)
        output_1 = torch.bmm(o1, o2)
        nepochs_in_x = len(x_)
        output_2 = output_1.view(nepochs_in_x, BMM_FEATURES)
        result = self.a3(output_2)
        return result


x = torch.randn(EPOCHS, BATCH_SIZE, IN_FEATURES)
y = torch.ones(EPOCHS, OUT_FEATURES)
for e in range(EPOCHS):
    for b in range(BATCH_SIZE):
        x[e, b, :] = x[e, 0, :]
    y[e, 0] = x[e, 0, 0]
    y[e, 1] = x[e, 0, 2]

net = Net()

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

for i in range(EPOCHS):
    net.zero_grad()
    output = net(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(loss.item())

predict = torch.ones(1, BATCH_SIZE, IN_FEATURES)
for b in range(BATCH_SIZE):
    predict[0, b, 0] = .3
    predict[0, b, 1] = .4
    predict[0, b, 2] = 1 - (.3 * .3) - (.4 * .4)
temp1 = net.forward(predict)
print(f'{predict[0] = }, {temp1 = }')
# temp1 = net.a1(predict)
# temp2 = net.a2(temp1).transpose(BATCH_DIM, OUT_FEATURE_DIM)
# temp3 = net.a3(temp2)
pass
