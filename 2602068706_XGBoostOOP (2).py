#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, RobustScaler, OneHotEncoder

import warnings
warnings.filterwarnings('ignore')


# In[3]:


class preprocessingData:
    # constructor
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None
    
    # load data from file path
    def loadData(self):
        self.data = pd.read_csv(self.file_path)
    
    # create input and ouput data frames
    def createInputOutput(self, targetColumn):
        self.output_df = self.data[targetColumn]
        self.input_df = self.data.drop(targetColumn, axis=1)


# In[4]:


class preprocessingModel:
    # constructor
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.XGBModel()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5
    
    # train test split
    def trainTestSplit(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.input_data, self.output_data, 
                                                                                test_size=test_size, random_state=random_state)
    # binary encoding for gender
    def binaryEncoding(self):
        binaryEncoded = {'Gender': {"Male": 1, "Female": 0}}
        self.x_train.replace(binaryEncoded, inplace=True)
        self.x_test.replace(binaryEncoded, inplace=True)
        return self.x_train.head()
    
    # binary encoding for geography
    def geoEncode(self):
        geo_encoded= {"Geography": {'Germany': 2, "Spain":1, "France":0}}
        self.x_train.replace(geo_encoded, inplace=True)
        self.x_test.replace(geo_encoded, inplace=True)
        return self.x_train.head()
    
    # label encoding for non-numerical data
    def labelEncoding(self, columns):
        label_encoding = LabelEncoder()
        self.x_train[columns] = label_encoding.fit_transform(self.x_train[columns])
        self.x_test[columns] = label_encoding.fit_transform(self.x_test[columns])
        return self.x_train.head()
                                                            
    # create boxplot to check outliers 
    def outlierCheck(self, columns):
        boxplot = self.x_train.boxplot(column=[columns])
        plt.show()
    
    # calculate median from a specified column
    def medianCalculation(self, columns):
        return self.x_train[columns].median()
    
    # handling missing values
    def handlingMissValues(self, columns, number):
        self.x_train[columns].fillna(number, inplace=True)
        self.x_test[columns].fillna(number, inplace=True)
    
    # drop columns
    def dropCol(self, columns):
        self.x_train = self.x_train.drop(columns=columns)
        self.x_test = self.x_test.drop(columns=columns)
    
    # robust scaling
    def robustScaler(self):
        scaler = RobustScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.fit_transform(self.x_test)
        
    # create XGBoost Classifier model
    def XGBModel(self, gamma=0, max_depth=4, n_estimators=50):
        self.model = XGBClassifier(gamma=gamma, max_depth=max_depth, n_estimators=n_estimators)
    
    # model training with train data
    def modelTraining(self):
        self.model.fit(self.x_train, self.y_train)
        
    # test data predictions
    def testDataPrediction(self):
        self.y_predict = self.model.predict(self.x_test)
    
    # classification report
    def classificationReport(self):
        print("\nClassification Report\n")
        print(classification_report(self.y_test, self.y_predict, target_names=['0', '1']))
    
    # evaluate model performances with test data
    def evaluateModel(self):
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions)
    
    # saving the trained model to a file
    def saveModel(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)


# In[14]:


# load data
file_path = 'data_D.csv'
data = preprocessingData(file_path)
data.loadData()
data.createInputOutput('churn')
input_df = data.input_df
output_df = data.output_df


# In[15]:


# preprocessing
model = preprocessingModel(input_df, output_df)
model.trainTestSplit()
model.binaryEncoding()


# In[16]:


model.labelEncoding('Surname')
model.geoEncode()


# In[17]:


print(model.medianCalculation('CreditScore'))


# In[18]:


model.handlingMissValues('CreditScore', 659.0)
model.dropCol('Surname')
model.dropCol('id')
model.dropCol('Unnamed: 0')
model.dropCol('CustomerId')


# In[19]:


model.robustScaler()


# In[20]:


model.modelTraining()
print("Model Accuracy:", model.evaluateModel())
model.testDataPrediction()
model.classificationReport()


# In[21]:


model.saveModel('XGB_ModelOOP.pkl')

