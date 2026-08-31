import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris=load_iris()
x=iris.data
y=iris.target

scaler=StandardScaler()
x=scaler.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=42)
x_train=torch.tensor(x_train,dtype=torch.float32)
y_train=torch.tensor(y_train,dtype=torch.long)
x_test=torch.tensor(x_test,dtype=torch.float32)
y_test=torch.tensor(y_test,dtype=torch.long)


class deepnn(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, num_classes):
        super(deepnn, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, num_classes)


    def forward(self,x):
        out=self.fc1(x)
        out=self.relu1(out)
        out=self.fc2(out)
        out=self.relu2(out)
        out=self.fc3(out)
        return out


input_size=x.shape[1]
hidden1,hidden2 = 16,8
num_classes = 3
model = deepnn(input_size,hidden1,hidden2,num_classes)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

epochs=100
for epoch in range(epochs):
    outputs = model(x_train)
    loss= criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(epoch+1) %10==0:
        print(f"Epoch[{epoch+1}/{epochs}], loss:{loss.item():.4f}")


with torch.no_grad():
    test_outputs = model(x_test)
    _, predicted = torch.max(test_outputs.data, 1)
    accuracy= (predicted == y_test).sum().item()/y_test.size(0)
    print("\n classification accuracy on test data: {:.2f}%".format(accuracy*100))