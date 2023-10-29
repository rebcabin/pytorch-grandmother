
#
# config
#

DIM_FROM    = 3

DIM_TO      = 2

REM_DIM     = 1

TOTAL_VECS  = 10000

SPLITS      = [ 0.8, 0.2, 0.2 ]

SEED        = 42

DSET_SIZES  = [ 0.1, 1.0 ]

BATCH_SIZES = [ 10, 20, 50 ]

LR_GAMMAS   = [ (0.5, 0.3), (0.5, 0.7), (1.0, 0.3), (1.0, 0.7) ]

MAX_EPOCHS  = [ 10, 100, 250 ]

EXPORT_FILE = "data/projection_architecture_search.csv"

#
# imports
#
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pandas as pd

#
# create a bunch of candidate neural net architectures
#

class Linear1(nn.Module):
    '''Define a 1-layer neural network, no drop out, no activation'''
    def __init__(self):
        super(Linear1, self).__init__()
        self.fc = nn.Linear(DIM_FROM, DIM_TO) 
    def forward(self, x):
        output = self.fc(x)
        return output

class Linear1_Dropout(nn.Module):
    '''Define a 1-layer neural network, with dropout, no activation'''
    def __init__(self, rate):
        super(Linear1_Dropout, self).__init__()
        self.dropout = nn.Dropout(rate)
        self.fc = nn.Linear(DIM_FROM, DIM_TO) 
    def forward(self, x):
        x = self.dropout(x)
        output = self.fc(x)
        return output

class Linear1_RELU(nn.Module):
    '''Define a 1-layer neural network, no dropout, with RELU activation'''
    def __init__(self):
        super(Linear1_RELU, self).__init__()
        self.fc = nn.Linear(DIM_FROM, DIM_TO) 
    def forward(self, x):
        x = self.fc(x)
        output = F.relu(x) 
        return output

class Linear1_Sigmoid(nn.Module):
    '''Define a 1-layer neural network, no dropout, with sigmoid activation'''
    def __init__(self):
        super(Linear1_Sigmoid, self).__init__()
        self.fc = nn.Linear(DIM_FROM, DIM_TO) 
    def forward(self, x):
        x = self.fc(x)
        output = F.sigmoid(x) 
        return output

class Linear1_Dropout_RELU(nn.Module):
    '''Define a 1-layer neural network, with dropout, with RELU activation'''
    def __init__(self, rate):
        super(Linear1_Dropout_RELU, self).__init__()
        self.dropout = nn.Dropout(rate)
        self.fc = nn.Linear(DIM_FROM, DIM_TO)
    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        output = F.relu(x)
        return output

class Linear1_Dropout_Sigmoid(nn.Module):
    '''Define a 1-layer neural network, with dropout, with sigmoid activation'''
    def __init__(self, rate):
        super(Linear1_Dropout_Sigmoid, self).__init__()
        self.dropout = nn.Dropout(rate)
        self.fc = nn.Linear(DIM_FROM, DIM_TO)
    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        output = F.sigmoid(x)
        return output

class Linear2(nn.Module):
    '''Define a 2-layer neural network, no dropout, no activation'''
    def __init__(self, hidden):
        super(Linear2, self).__init__()
        self.fc1 = nn.Linear(DIM_FROM, hidden)
        self.fc2 = nn.Linear(hidden, DIM_TO)
    def forward(self, x):
        x = self.fc1(x)
        output = self.fc2(x)
        return output

class Linear2_Dropout(nn.Module):
    '''Define a 2-layer neural network, dropout all layers, no activation'''
    def __init__(self,rate, hidden):
        super(Linear2_Dropout, self).__init__()
        self.dropout1 = nn.Dropout(rate)
        self.fc1 = nn.Linear(DIM_FROM, hidden)
        self.dropout2 = nn.Dropout(rate)
        self.fc2 = nn.Linear(hidden, DIM_TO)
    def forward(self, x):
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        output = self.fc2(x)
        return output

class Linear2_RELU1(nn.Module):
    '''Define a 2-layer neural network, no dropout, RELU activation first layer'''
    def __init__(self,hidden):
        super(Linear2_RELU1, self).__init__()
        self.fc1 = nn.Linear(DIM_FROM, hidden)
        self.fc2 = nn.Linear(hidden, DIM_TO)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        output = self.fc2(x)
        return output

class Linear2_RELU2(nn.Module):
    '''Define a 2-layer neural network, no dropout, RELU activation all layers'''
    def __init__(self,hidden):
        super(Linear2_RELU2, self).__init__()
        self.fc1 = nn.Linear(DIM_FROM, hidden)
        self.fc2 = nn.Linear(hidden, DIM_TO)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.relu(x)
        return output

def train(model, device, train_loader, optimizer, epoch):
    '''Define the train function'''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()

def validate(model, device, validate_loader):
    '''Define the validation function'''
    model.eval()
    validate_loss = 0
    with torch.no_grad():
        for data, target in validate_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            validate_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss

    validate_loss /= len(validate_loader.dataset)
    return validate_loss

def test(model, device, test_loader):
    '''Define the test function'''
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    return test_loss

def architecture_search():
    '''Main function to search the architecture space for best model'''

    # don't overwrite any existing file
    if os.path.exists(EXPORT_FILE):
        raise Exception("The file %s already exists.  Please move that file if your intention is to redo the entire architecture search." % EXPORT_FILE)
        
    #
    # set ran seeds for reproducibility
    #
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # choose accelerator if any
    device = torch.device("cpu")

    #
    # initiaize dataset in numpy
    #
    dset_in = np.random.rand( TOTAL_VECS, DIM_FROM )
    dset_out = np.array( [ np.array( list(el[0:REM_DIM]) + list(el[REM_DIM+1:]) ) for el in dset_in ] )
    # sanity check
    print("dataset sanity check...")
    print(dset_in.shape, dset_in.dtype, dset_in[0].dtype)
    print(dset_out.shape, dset_out.dtype, dset_out[0].dtype)
    print(dset_in[0], dset_out[0])
    print(dset_in[-1], dset_out[-1])
    print()

    # use sci-kit to create train/validate/test splits
    x_train, x_test, y_train, y_test = train_test_split(dset_in, dset_out, test_size=0.4)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)
    # sanity check
    print("tr/val/test split sanity check:", x_train.shape, x_val.shape, x_test.shape)
    print()

    #
    # initialize the architecture and hyperparameter search
    #
    all_nets = \
        [   
            ( Linear1,                ()       ),
            ( Linear1_Dropout,        (0.1,)   ),
            ( Linear1_Dropout,        (0.5,)   ),
            ( Linear1_Dropout,        (0.9,)   ),
            ( Linear1_RELU,           ()       ),
            ( Linear1_Sigmoid,       ()       ),
            ( Linear1_Dropout_RELU,   (0.1,)   ),
            ( Linear1_Dropout_RELU,  (0.5,)   ),
            ( Linear1_Dropout_RELU,   (0.9,)   ),
            ( Linear1_Dropout_Sigmoid,(0.1,)   ),
            ( Linear1_Dropout_Sigmoid,(0.5,)   ),
            ( Linear1_Dropout_Sigmoid,(0.9,)   ),
            ( Linear2,                (5,)      ),
            ( Linear2,                (10,)     ),
            ( Linear2_Dropout,        (0.1,5)   ),
            ( Linear2_Dropout,        (0.5,5)   ),
            ( Linear2_Dropout,        (0.9,5)   ),
            ( Linear2_Dropout,        (0.1,10)  ),
            ( Linear2_Dropout,        (0.5,10)  ),
            ( Linear2_Dropout,        (0.9,10)  ),
            ( Linear2_RELU1,          (5,)      ),
            ( Linear2_RELU1,          (10,)     ),
            ( Linear2_RELU2,          (5,)      ),
            ( Linear2_RELU2,          (10,)     )
    ] 
   
    # emit header 
    print('{: <30}{: <20}{: <10}{: <10}{: <10}{: <10}{: <10}{:<20}'.format(\
            "model", "mod parms", "tr samps", "bsize", "lr", "gamma", "epochs", "avg test loss"))

    # capture all experiment results for export later
    results = []

    # 
    # iterate through the hyperparameter space
    #
    for net in all_nets:

        # create the model with parameters
        arch = net[0]
        parms = net[1]
        model = arch(*parms).to(device)

        # iterate dataset set sizes
        for frac in DSET_SIZES:
   
            # 
            # take portion of master numpy dataset and coax into pytorch tensors 
            #

            x_train_tensors = torch.Tensor(x_train[0:int(frac*len(x_train))]) 
            y_train_tensors = torch.Tensor(y_train[0:int(frac*len(y_train))])
            train_tensors = TensorDataset(x_train_tensors,y_train_tensors)

            x_validate_tensors = torch.Tensor(x_val[0:int(frac*len(x_val))])
            y_validate_tensors = torch.Tensor(y_val[0:int(frac*len(y_val))])
            validate_tensors = TensorDataset(x_validate_tensors,y_validate_tensors)

            x_test_tensors = torch.Tensor(x_test[0:int(frac*len(x_test))])
            y_test_tensors = torch.Tensor(y_test[0:int(frac*len(y_test))])
            test_tensors = TensorDataset(x_test_tensors,y_test_tensors)
       
            # iterate batch sizes
            for batch_size in BATCH_SIZES:

                # create the batch loaders
                train_loader = DataLoader(train_tensors,batch_size=batch_size )
                validate_loader = torch.utils.data.DataLoader(validate_tensors, batch_size=batch_size)
                test_loader = torch.utils.data.DataLoader(test_tensors, batch_size=batch_size)

                # iterate learning rate/gamma pairs
                for opt_parm in LR_GAMMAS:
                    learning_rate, gamma = opt_parm

                    # create optimizer and scheduler
                    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
                    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

                    # iterate different number of training epochs
                    for max_epochs in MAX_EPOCHS:

                        # the training loop for this model
                        for epoch in range(1, max_epochs + 1):
                            train(model, device, train_loader, optimizer, epoch)
                            validate(model, device, validate_loader)
                            scheduler.step()
                
                        # test the model
                        avg_test_loss = test(model, device, test_loader)

                        # capture result
                        result = {  'model':model.__class__.__name__, 'mparms':str(parms), 'trsamples':int(frac*len(x_train)),\
                                    'bsize':batch_size, 'lr':learning_rate, 'gamma':gamma, 'maxepochs':max_epochs, 'avgtestloss': avg_test_loss } 
                        results.append(result)

                        # print summary   
                        print('{: <30}{: <20}{: <10}{: <10}{: <10}{: <10}{: <10}{:<20.4f}'.format(\
                            result['model'], result['mparms'], result['trsamples'], result['bsize'], result['lr'], result['gamma'],\
                            result['maxepochs'], result['avgtestloss']))

                        # export interim results
                        df = pd.DataFrame(results)
                        df.to_csv(EXPORT_FILE, sep='\t') 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-", "--search", action='store_true', help="perform architecture search")
    args = parser.parse_args()
    
    if args.search:
        architecture_search()
    else:
        print("Nothing to do.") 
