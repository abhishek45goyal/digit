#!/usr/bin/env python
# coding: utf-8

# In[24]:


from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# In[5]:


mnist = fetch_openml('mnist_784')


# In[6]:


x, y = mnist['data'], mnist['target']

some_digit = x.to_numpy()[36001]
some_digit_image = some_digit.reshape(28, 28)  # let's reshape to plot it

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
           interpolation='nearest')
plt.axis("off")
plt.show()


# In[13]:


x_train, x_test = x[:60000], x[6000:70000]
y_train, y_test = y[:60000], y[6000:70000]

shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train.[shuffle_index],y_train.[shuffle_index]

# Creating a 2-detector
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_2 = (y_train == '2')
y_test_2 = (y_test == '2')


# In[14]:


x_train, x_test = x[:60000], x[6000:70000]
y_train, y_test = y[:60000], y[6000:70000]


# In[15]:



shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train.[shuffle_index],y_train.[shuffle_index]


# In[16]:


x_train, x_test = x[:60000], x[6000:70000]
y_train, y_test = y[:60000], y[6000:70000]

shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index],y_train[shuffle_index]

# Creating a 2-detector
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_2 = (y_train == '2')
y_test_2 = (y_test == '2')


# In[19]:



# Assuming you have already defined x and y

# Splitting the data into train and test sets
x_train, x_test = x[:60000], x[60000:70000]
y_train, y_test = y[:60000], y[60000:70000]

# Shuffling the training data
shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train.[shuffle_index], y_train.[shuffle_index]

# Creating a 2-detector
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_2 = (y_train == 2)
y_test_2 = (y_test == 2)


# In[20]:


x_train, x_test = x[:60000], x[60000:70000]
y_train, y_test = y[:60000], y[60000:70000]


# In[23]:


import numpy as np
shuffle_index=np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]


# In[25]:


# Assuming you have already defined x and y

# Splitting the data into train and test sets
x_train, x_test = x[:60000], x[60000:70000]
y_train, y_test = y[:60000], y[60000:70000]

# Shuffling the training data
shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

# Creating a 2-detector
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_2 = (y_train == 2)
y_test_2 = (y_test == 2)


# In[26]:


import numpy as np
import pandas as pd  # Assuming you are working with Pandas DataFrames

# Assuming you have already defined x_train and y_train as Pandas DataFrames
# You can add some error-checking before shuffling

# Check the shape of x_train and y_train
print("Shape of x_train:", x_train.shape)
print("Shape of y_train:", y_train.shape)

# Ensure that the number of rows matches
if x_train.shape[0] == y_train.shape[0]:
    # Shuffling the training data
    shuffle_index = np.random.permutation(len(x_train))
    x_train = x_train.iloc[shuffle_index]
    y_train = y_train.iloc[shuffle_index]
else:
    print("Error: Number of rows in x_train and y_train don't match.")

# Creating a 2-detector
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_2 = (y_train == 2)
y_test_2 = (y_test == 2)


# In[27]:


# Train a logistic regression classifier
clf = LogisticRegression(tol=0.1)
clf.fit(x_train, y_train_2)
example = clf.predict([some_digit])
print(example)


# In[28]:


# Cross Validation
a = cross_val_score(clf, x_train, y_train_2, cv=3, scoring="accuracy")
print(a.mean())


# In[ ]:




