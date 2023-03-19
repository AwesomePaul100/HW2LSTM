#!/usr/bin/env python
# coding: utf-8

# In the assignment, you are asked to practice how to use a 2-layered Long Short Term Memory (LSTM) i.e. a recurrent neural network (RNN) architecture to generate a character level model to predict the next upcoming note in a music sequence.

# In[1]:


#Imports
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
import math
import requests
import matplotlib.pyplot as plt

# Get the txt file.
#url = “https://raw.githubusercontent.com/cedricdeboom/character-level-rnn-datasets/master/datasets/music.txt”
url = 'https://raw.githubusercontent.com/cedricdeboom/character-level-rnn-datasets/master/datasets/music.txt'
resp = requests.get(url)
with open('music.txt', 'wb') as f:
    f.write(resp.content)


# In[2]:


# All datasets represented as strings. 
#Therfore, we must process the characters.

data = open('music.txt', 'r').readlines()
data1 = [list(x.strip().split(' ') + ['\n']) for x in data]
print(data1[0][1004],(data[0]),data[1])
print(len(data1))


# In[3]:


from sklearn.preprocessing import normalize
masterlist = []
sublist = []
inputlist = []
target = []
squencelength = 5
print(data1[0][1])
for i in range(2):
    #print(i, len(data[i]))
    length = (data1[i].index('\n'))
    #print(data1[0].index('\n'))
    # Each section is made into a subsection list.
    for j in range(length):
        #print("counter value",j)
        # Making it into an int value.
        sublist.append(int(data1[i][j]))
    for j in range(len(sublist)):
        if (len(sublist) - j) > squencelength + 1:
            for p in range(squencelength):
                inputlist.append((sublist[j+p]))
            # Increment counter.
            #print(j)
            #j = j + 100
            target.append((sublist[j+squencelength]))
            # Normalize data.
            #norm = [float(i)/max(inputlist) for i in inputlist]
            masterlist.append(inputlist)
            inputlist = []
    print(len(sublist),masterlist,target)
    sublist = []


# In[4]:


print(len(masterlist),masterlist[2],target[2])


# ## In the following cells each sequence will utilize the 703 sequences in the music file, and the model must predict the last note of each sequence. Therefore, we will Train 80 percent of the data and Test on the remainder. The Hope is that the model is able to learn from the songs in the other sequences and perform a well prediction

# In[5]:


# 80 percent of number of sequences (503) if sequence length is 500
print(503*.8)


# In[7]:


# Define the input data
input_data = []
#print(masterlist)
print(len(target))
# Splitting into Training-Test Sets (80/20)

from sklearn.model_selection import train_test_split
x_train_original, x_test_original, y_train_original, y_test_original = train_test_split(masterlist, target,test_size = 0.2, random_state = 42)

#x_train_original = masterlist[0:402:1]
#y_train_original = target[0:402:1]
#x_test_original = masterlist[403:503:1]
#y_test_original = target[403:503:1]
print(len(x_train_original),len(x_test_original),len(y_train_original))

# Make the input data to PyTorch tensors.
x_train = torch.tensor(x_train_original).unsqueeze(2).float()

# Define the target data to train.
target_data = []
for i in y_train_original:
    target_data.append([i])
print(len(target_data))

# Define the target data to test
target_data_test = []
for i in y_test_original:
    target_data_test.append([i])
#print(target_data_test)

# Make the target data to PyTorch tensors
y_train = torch.tensor(target_data).float()

# Make the input data to PyTorch tensors.
x_test = torch.tensor(x_test_original).unsqueeze(2).float()
# Make the input data to PyTorch tensors.
y_test = torch.tensor(target_data_test).unsqueeze(2).float()

# Make the input data to PyTorch tensors.
#x = torch.tensor(x_train).unsqueeze(2).float()

print("x_test:", x_test,"y_test:", y_test, "x_train:\n", x_train,"y_train:\n", y_train)
print("x_test length:", len(x_test), "x_train length:", len(y_train))
# Make the target data to PyTorch tensors.
#y = torch.tensor(y_train).float()


# In[40]:


import time
# Set the LSTM model.
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        # We first call the super initializer.
        super(LSTMModel, self).__init__()
        # set hidden.
        self.hidden_size = hidden_size
        # set num layers.
        self.num_layers = num_layers
        # The first layer.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # The last layer is linear that outputs the predicted value.
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_t = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_t = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h_t, c_t))
        out = self.fc(out[:, -1, :])
        return out



# Set the hyperparameters for the model.
input_size = 1
hidden_size = 64
# Increased the number of layers to three to hopefully get better accuracy.
num_layers = 3
output_size = 1

# Instantiate the LSTM model.
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Set the loss function and the optimizer.
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model.
num_epochs = 70000
st = time.time()
for epoch in range(num_epochs):
    # The forward pass.
    outputs = model(x_train)

    # Compute the loss after the forward pass.
    loss = criterion(outputs, y_train)

    # Now Backward pass and imporve (optimize) the model.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print every 100 Epoch.
    if (epoch+1) % 100 == 0:
        end_time = time.time()
        print('Epoch [',epoch+1,'/',num_epochs,'], and Loss:', loss.item(), "time elapsed to do 100 epochs:", end_time - st)
        # Reset the start time.
        st = time.time()


# Predict the last note in the given sequence.
import numpy as np
for i in range(len(x_test_original)):  
    test_input = torch.tensor([x_test_original[i]]).unsqueeze(2).float()
    #print(test_input)
    predicted_output = model(test_input)
    print("Predicted value: ",predicted_output.item(), "\nThe actual value: ", y_test_original[i])
    MSE = np.square(np.subtract(y_test_original[i],predicted_output.item())).mean() 
    print("The Mean Square Error is: ", MSE)
    RMSE = math.sqrt(MSE)
    print("The Root Mean Square Error is: ", RMSE)


# Count the accuracy.
print("\nNow calculate the accuracy\n")
countright = 0
for i in range(len(x_test_original)):  
    test_input = torch.tensor([x_test_original[i]]).unsqueeze(2).float()
    #print(test_input)
    predicted_output = model(test_input)
    print("Predicted value: ",predicted_output.item(), "\nThe actual value: ", y_test_original[i])
    MSE = np.square(np.subtract(y_test_original[i],predicted_output.item())).mean() 
    print("The Mean Square Error is: ", MSE)
    RMSE = math.sqrt(MSE)
    print("The Root Mean Square Error is: ", RMSE)
    if round(predicted_output.item()) == y_test_original[i]:
        countright = 1 + countright
print("accuracy:", countright/len(x_test_original))


# Count the accuracy.
countright = 0
predicted_values= []
for i in range(len(x_test_original)):
    #print(x_test_original[i],y_test_original[i])
    test_input = torch.tensor([x_test_original[i]]).unsqueeze(2).float()
    #print(test_input)
    predicted_output = model(test_input)
    predicted_values.append(int(predicted_output.item()))
   # print("Predicted value: ",predicted_output.item(), "\nThe actual value: ", y_test_original[i])
    MSE = np.square(np.subtract(y_test_original[i],predicted_output.item())).mean() 
    #print("The Mean Square Error is: ", MSE)
    RMSE = math.sqrt(MSE)
    #print("The Root Mean Square Error is: ", RMSE)
    if (round(predicted_output.item()) + 1 == y_test_original[i]) or (round(predicted_output.item()) + 1 == y_test_original[i]):
        countright = 1 + countright

# Print scatter plot.
# Red for our predicted, and blue for the true(target) values.
plt.scatter(range(len(y_test_original)),predicted_values,c='r')
plt.scatter(range(len(y_test_original)),y_test,c='b')
plt.show()

print("accuracy with some deviaton:", countright/len(x_test_original))

print("test",len(x_test),"train",len(x_train))



