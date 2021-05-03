#!/usr/bin/env python
# coding: utf-8

# In[1]:




import keras
import glob
import pandas as pd
import numpy as np
import timeit
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Save Model Using Pickle
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

# In[2]:


cross1=pd.read_csv('oasis_longitudinal.csv')
print(cross1)
cross1 = cross1.fillna(method='ffill')
cross2=pd.read_csv('oasis_cross-sectional.csv')
cross2 = cross2.fillna(method='ffill')


# In[3]:


cross1.head()
print(cross1)

# In[4]:


cross2.head()
print(cross2)

# In[5]:


cross1.info()


# In[6]:


cross2.info()


# In[7]:


cross1.drop(['MRI ID'], axis=1, inplace=True)
cross1.drop(['Visit'], axis=1, inplace=True)


# In[8]:


#cdr=cross1["CDR"]
cross1['CDR'].replace(to_replace=0.0, value='A', inplace=True)
cross1['CDR'].replace(to_replace=0.5, value='B', inplace=True)
cross1['CDR'].replace(to_replace=1.0, value='C', inplace=True)
cross1['CDR'].replace(to_replace=2.0, value='D', inplace=True)
print(cross1)

# In[9]:


from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
for x in cross1.columns:
    f = LabelEncoder()
    cross1[x] = f.fit_transform(cross1[x])


# In[10]:


cross1.head()
print(cross1)

# In[11]:


train, test = train_test_split(cross1, test_size=0.3)


# In[12]:


X_train = train[['M/F', 'Age', 'EDUC', 'SES',  'eTIV', 'ASF']]
print(X_train)
y_train = train.CDR
print(y_train)
X_test = test[['M/F', 'Age', 'EDUC', 'SES',  'eTIV',  'ASF']]
y_test = test.CDR


# In[13]:


from sklearn.preprocessing import StandardScaler

# Define the scaler
scaler = StandardScaler().fit(X_train)
print(scaler)
# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)


# In[14]:


y_train=np.ravel(y_train)
X_train=np.asarray(X_train)

y_test=np.ravel(y_test)
X_test=np.asarray(X_test)


# In[15]:


import tensorflow as tf
from sklearn import metrics
X_FEATURE = 'x'  # Name of the input feature.
feature_columns = [
      tf.feature_column.numeric_column(
          X_FEATURE, shape=np.array(X_train).shape[1:])]


classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[35,70, 35], n_classes=4)

  # Train.
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={X_FEATURE: X_train}, y=y_train, num_epochs=100, shuffle=False)
classifier.train(input_fn=train_input_fn, steps=1000)

  # Predict.
test_input_fn = tf.estimator.inputs.numpy_input_fn(x={X_FEATURE: X_test}, y=y_test, num_epochs=1, shuffle=False)
predictions = classifier.predict(input_fn=test_input_fn)
y_predicted = np.array(list(p['class_ids'] for p in predictions))

y_predicted = y_predicted.reshape(np.array(y_test).shape)
print("test data")
print(y_test[:10])
print("predicted data")
print(y_predicted[:10])



scores = classifier.evaluate(input_fn=test_input_fn)
print('Accuracy: {0:f}'.format(scores['accuracy']))


# In[16]:


y_train


# In[17]:


cross1.head()


# In[18]:


cross2.head()


# In[19]:


for x in cross2.columns:
    f = LabelEncoder()
    cross2[x] = f.fit_transform(cross2[x])


# In[20]:


df = pd.concat([cross1,cross2])


# In[21]:


df = df.fillna(method='ffill')
df.head()


# In[22]:


train, test = train_test_split(cross1, test_size=0.3)
X_train1 = train[['ASF', 'Age', 'EDUC', 'Group',  'Hand', 'M/F','MMSE','MR Delay','SES','eTIV','nWBV']]
y_train1 = train.CDR
X_test1 = test[['ASF', 'Age', 'EDUC', 'Group',  'Hand', 'M/F','MMSE','MR Delay','SES','eTIV','nWBV']]
y_test1 = test.CDR


# In[23]:


from sklearn.preprocessing import StandardScaler

# Define the scaler
scaler = StandardScaler().fit(X_train1)

# Scale the train set
X_train1 = scaler.transform(X_train1)

# Scale the test set
X_test1 = scaler.transform(X_test1)


# In[24]:


y_train1=np.ravel(y_train1)
X_train1=np.asarray(X_train1)

y_test1=np.ravel(y_test1)
X_test1=np.asarray(X_test1)


# In[25]:


X_train1


# In[26]:


import tensorflow as tf
from sklearn import metrics
X_FEATURE = 'x'  # Name of the input feature.
feature_columns = [
      tf.feature_column.numeric_column(
          X_FEATURE, shape=np.array(X_train1).shape[1:])]

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[35,70,35], n_classes=4)

  # Train.
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={X_FEATURE: X_train1}, y=y_train1, num_epochs=100, shuffle=False)
classifier.train(input_fn=train_input_fn, steps=1000)
#save_model(classifier, "model.hdf5", overwrite=True, include_optimizer=True)
  # Predict.
test_input_fn = tf.estimator.inputs.numpy_input_fn(x={X_FEATURE: X_test1}, y=y_test1, num_epochs=1, shuffle=False)
predictions = classifier.predict(input_fn=test_input_fn)
y_predicted = np.array(list(p['class_ids'] for p in predictions))
y_predicted = y_predicted.reshape(np.array(y_test1).shape)
print(y_predicted[:10])




  # Score with tensorflow.
score1 = classifier.evaluate(input_fn=test_input_fn)
print('Accuracy of DNN: {0:f}'.format(scores['accuracy']))


# In[27]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import feature_selection
from sklearn import model_selection
classifier = DecisionTreeClassifier(max_depth=5)
history = classifier.fit(X_train1, y_train1)
print("-------------------------------------------------------------------------------------------")

prediction = classifier.predict(X_test1)
print(prediction[:10])
a=classifier.score(X_train1, y_train1)
model = DecisionTreeClassifier()
model.fit(X_train1, y_train1)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test1, y_test1)
print(result)
1
print("Train accuracy:",a)
b=classifier.score(X_test1, y_test1)
print("Test accuracy:",b)


# In[28]:


train=[score1,a]
train1=["DNN","DecisionClassifier"]
print(train)
print(train1)




# fig, ax = plt.subplots()
# ax.scatter(train1, train)
# n = [score,a]
# for i, txt in enumerate(n):
#     ax.annotate(txt, (train1[i], train[i]))
# plt.ylabel('accuracy')
# plt.xlabel('Classfier')
# plt.show()









