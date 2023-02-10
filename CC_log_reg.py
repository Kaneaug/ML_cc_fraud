import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset into pandas
credit_card_data = pd.read_csv(r'C:\Users\kanem\Documents\ML_cc_fraud\data\creditcard.csv')

# Data exploration
credit_card_data.head()
credit_card_data.tail()
credit_card_data.info()

# Find missing values
credit_card_data.isnull().sum()

#Check balance of fraud vs. legitimate purchases
credit_card_data['Class'].value_counts()

# Fraud imbalance 492/284315*100 = 0.17304750013189596

# Seperate into legitimate and fraud classes
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# analyze data further
legit.Amount.describe()
fraud.Amount.describe()

legit.shape
fraud.shape

credit_card_data.groupby('Class').mean()

#Building a sample dataset to test distribution balance by undersampling

legit_sample = legit.sample(n=492)

new_dataset = pd.concat([legit_sample, fraud], axis=0)
new_dataset.head()
new_dataset['Class'].value_counts()
new_dataset.groupby('Class').mean()

#Split into features and target and test and training data

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y,random_state=2)
X.shape, X_train.shape, X_test.shape

# Training data and Regression model 
model = LogisticRegression()
model.fit(X_train,Y_train)

# Evaulating data and accuracy

X_train_prediction = model.predict(X_train)
train_data_score = accuracy_score(X_train_prediction,Y_train)

print('Accuracy of Training score: ', train_data_score)

X_test_prediction = model.predict(X_test)
test_data_score = accuracy_score(X_test_prediction, Y_test)
print('Accuracy of Test score: ', test_data_score)