#!/usr/bin/env python
# coding: utf-8

# In[81]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd 


# In[82]:


class Model(nn.Module):
    def __init__(self, in_features = 4, h1=8, h2=9, out_features=3):
        super().__init__() 
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        
        return x


# In[83]:


torch.manual_seed(41)
model = Model()


# In[84]:


#obtain the csv / dataset
url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df = pd.read_csv(url)


# In[85]:


#change species name to number values
my_df["species"] = my_df["species"].replace("setosa", 0.0)
my_df["species"] = my_df["species"].replace("versicolor", 1.0)
my_df["species"] = my_df["species"].replace("virginica", 2.0)
my_df


# In[86]:


#Train test 
x = my_df.drop('species', axis = 1)
y = my_df['species']


# In[87]:


#convert to numpy arrays
x = x.values
y = y.values
print(x)


# In[88]:


from sklearn.model_selection import train_test_split


# In[89]:


#train test split
x_train, x_test, y_train, y_test, = train_test_split(x, y, test_size=0.2, random_state = 41)


# In[90]:


x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


# In[91]:


#Set criterion of model to measure error
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# In[92]:


#Train model
#100 epochs
epoch = 100
losses = []
for i in range(epoch):
    y_pred = model.forward(x_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.detach().numpy())
    
    #print every 10th epoch
    if i % 10 == 0:
        print(f'Epoch: {i} and loss: {loss}')
    
    #back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# In[93]:


# graph
plt.plot(range(epoch), losses)
plt.ylabel("loss/error")
plt.xlabel("Epoch")


# In[94]:


#evaluate model
#turn back propagation off
with torch.no_grad():
    y_eval = model.forward(x_test) 
    loss = criterion(y_eval, y_test)
    print(loss)


# In[95]:


correct = 0 
with torch.no_grad():
    for i, data in enumerate(x_test):
        y_val = model.forward(data)
  
        print(f'{i+1}.) {str(y_val)} \t Guess: {y_test[i]} \t Real: {y_val.argmax().item() } ')
        
        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(f'Num of Correct: {correct}')


# In[ ]:




