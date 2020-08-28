import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import csv


#model definition with 3 Hidden Layers with Relu and sigmoid activations. 

class Net(nn.Module):
    
    # Constructor
    def __init__(self, D_in, H1,H2,H3,D_out):
        super(Net, self).__init__()

        self.linear1 = nn.Linear(D_in, H1)
        #hidden layer 
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        #output layer 
        self.linear4 = nn.Linear(H3, D_out)

    # Prediction    
    def forward(self, x):
        x = torch.relu(self.linear1(x))  
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        return x

# function to calculate accuracy
def accuracy(y, yhat):
    yhat = yhat.cpu().detach().numpy()
    acc = np.argmax(yhat)
    return (y==acc).detach().cpu().numpy()


# helper function-1 used for reading any input file even if it's not a csv
def read_tsv(filepath): 
    data_testList = []
    
    file1 = open(filepath, 'r') 
    data_testList = file1.readlines()
    return np.asarray(data_testList)

# helper function-2 used for reading any input file even if it's not a csv
def split_line(train_raw):
    label = []
    features = []
    a = []
    for line in  train_raw :
        temp = line.split('\n')
        a.append(temp[0])      
    for line in a:
        line = line.split(',')
        label.append(line[-1])
        features.append(line[1:])
        
    return np.asarray(label),np.asarray(features)

#Function to train
    
def train(device, data_set,test_data_set, model, criterion, train_loader,test_loader, optimizer, epochs=5):

    for epoch in range(epochs):         
        total=0                                         
        ACC = []
        print("\n")
        print("epoch: "+ str(epoch))        #printing epoch count
        for x, y in train_loader:           
            x =x.to(device)                   #converting variables to cuda from cpu for faster processing 
            y =y.to(device)           
            optimizer.zero_grad()           #optimizer resetting the gradient
            yhat = model(x.float())         # function call for each batch of data
            loss = criterion(yhat, y.long())    #CrossEntropyLoss between target label and predicted values
            loss.backward()         
            optimizer.step()                        
            total+=loss.item()              #cumulative loss
            acc = accuracy(y, yhat)         #accuracy calculation for each iteration.          
            ACC.append(np.sum(acc))         #list to keep track of accuracy for each epoch
                      
        print("loss: " + str(total))
        print(str(np.sum(ACC))+ " / "+ str(len(data_set)) + "  Percentage: "+ str(np.sum(ACC)/len(data_set)))

        #call to the validation function (checking performance for every iteration )
       # validation(data_set, test_data_set, model, criterion, test_loader, epochs=1)
    


 #function used for validation before testing  to make sure the model didn't overfit the data and has lesser variance.
def validation(device, data_set,test_data_set, model, criterion,train_loader, epochs=1):

    ACC = []
    total=0
    for x, y in train_loader:           
        x =x.to(device)          
        y =y.to(device)
        yhat = model(x.float())

        loss = criterion(yhat, y)

        #cumulative loss 
        total+=loss.item()
        acc = accuracy(y, yhat)
        ACC.append(acc.item())

    print("loss_test: " + str(total))
    print("correct: " + str(np.sum(ACC))+ "/" + str(len(test_data_set)) + "  Percentage: "+ str(np.sum(ACC)/len(test_data_set)))

#final test_data evaluation
def evaluation(model,criterion,test_loader):
    with open('test_label1.csv', mode='w') as file:     #opening a file
        id = np.arange(7201, 12000)      
        for id_,x in zip(id,test_loader):        
          #  x = x.cuda()
            
            print()
            predicted = model(x.float())   #obtaining the probabilities of predicted labels
            
            label = predicted.cpu().detach().numpy()            #converting to numpy for getting the max argument
            label = np.argmax(label)                    # finding the predicted label for the data input.
            
            writer = csv.writer(file, delimiter=',', lineterminator='\n') #adjustments for delimiting and terminating.
            writer.writerow([id_,label])            #writing onto the csv file.
        
#function to create the dataset with labels (inherited from the Dataset class)
class Custom_dataset(Dataset):

    def __init__(self,x,y):
      self.x=x
      self.y=y
    def __getitem__(self, index):    
        return torch.from_numpy(self.x[index]),torch.tensor(self.y[index]).long()
    def __len__(self):
        return self.x.shape[0]

#function to create the test data without labels to be used for  testing. 
class Custom_dataset_test(Dataset):

    def __init__(self,x):
      self.x=x
    def __getitem__(self, index):    
        return torch.from_numpy(self.x[index])
    def __len__(self):
        return self.x.shape[0]

#main for program execution
if __name__ == '__main__':
    
    train_input = 'train.csv'        
    
    #snippet to read the input csv using numpy and the helper functions
    
    #removing the first row and converting the last column into labels.
    
    train_array = read_tsv(train_input)                                 
    labels_train_main, features_train_main = split_line(train_array)
    features_train_main = features_train_main[1:,:-1]
    features_train_main = features_train_main.astype(np.float)
    labels_train_main = labels_train_main[1:]
    labels_train_main = labels_train_main.astype(np.int)    
    
# =============================================================================
#     labels_train = labels_train_main
#     features_train = features_train_main
# =============================================================================
    
    
    #data split for validation.
    # =============================================================================
    # labels_test = labels_train_main[5001:] 
    # features_test = features_train_main[5001:]
    # 
    # =============================================================================
    
    #creation of dataset.
    
    data_set = Custom_dataset(features_train_main,labels_train_main)
    # =============================================================================
    # val_data_set = Custom_dataset(features_test,labels_test)
    # 
    # =============================================================================
    
    
    #model initialisation
    model = Net(250, 128, 64, 32, 5)
    
    #gpu check
    cuda = torch.cuda.is_available()
    device = torch.device( 'cuda' if cuda else 'cpu' )
    model.to(device)
    
    learning_rate = 0.001
     #loss criterion - CEL for multiclass classification.
    criterion = nn.CrossEntropyLoss()
    
    #stochastic gradient descent  for optimisation with learning rate, L2 Regulariser as penalty to avoid overfitting.
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.05)
    
    #creating train and validation loader.
    train_loader = DataLoader(dataset=data_set, batch_size=1,shuffle = True)
    
    val_loader = 0
    val_data_set = []
    # =============================================================================
    # val_loader = DataLoader(dataset=val_data_set, batch_size=1 , shuffle = True)
    # 
    # =============================================================================
    
    #call to begin training. 
    LOSS12 = train(device,data_set,val_data_set, model, criterion, train_loader,val_loader, optimizer, epochs=20 )
    
    #saving the model after training.
    torch.save(model, 'weights.pt')
    
    
    
    # loaded the weights back to make sure it works.
    #testing after saving and loading the weights (check for saving )
    
    model = Net(250, 128, 64, 32, 5)
    checkpoint  = torch.load('weights.pt')
    model.load_state_dict(checkpoint.state_dict())
    model.eval()
    
    
    #reading the test dataset 
    test_input = 'test.csv'        
    test_array = read_tsv(test_input)
    
    labels_test_main, features_test_main = split_line(test_array)
    features_test_main = features_test_main[1:]
    features_test_main = features_test_main.astype(np.float)
    
    #test dataset for loader
    test_data_set = Custom_dataset_test(features_test_main) 
    #testloader
    test_loader = DataLoader(dataset=test_data_set, batch_size=1,shuffle = True)
    #evaluation and writing to test_label file
    evaluation(model,criterion,test_loader)




  






