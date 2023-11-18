# Titanic_Survival_Prediction
# Importing all the necessary liblaries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Data Collection
# Load the dataframe from csv file to pandas dataframe
titanic_data = pd.read_csv('/content/tested.csv')

#printing the first 5 rows of the dataframe
titanic_data.head()

# number of rows and Columns
titanic_data.shape

# getting some informations about the data
titanic_data.info()

# check the number of missing values in each column
titanic_data.isnull().sum()

# drop the "Cabin" column from the dataframe
titanic_data = titanic_data.drop(columns='Cabin', axis=1)

# replacing the missing values in "Age" column with mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)

# finding the mode value of "Embarked" column
print(titanic_data['Embarked'].mode())
print(titanic_data['Embarked'].mode()[0])

# replacing the missing values in "Embarked" column with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# check the number of missing values in each column
titanic_data.isnull().sum()

# getting some statistical measures about the data
titanic_data.describe()

# finding the number of people survived and not survived
titanic_data['Survived'].value_counts()
# Data Visualization
sns.countplot(x='Survived', data=titanic_data)
plt.title('Survival Count')
plt.show()

# making a count plot for 'Sex' column
sns.countplot(x='Survived', data=titanic_data)
plt.title('Survival Count')
plt.show()

# Getting number of surviver gender wise
sns.countplot(x='Sex', hue='Survived', data=titanic_data)
plt.title('Survival Count by Sex')
plt.show()
# making a count plot for "Pclass" column
sns.countplot(x='Sex', hue='Survived', data=titanic_data)
plt.title('Survival Count by Sex')
plt.show()

# Seperating features & target 
X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = titanic_data['Survived']

print(X)
print(Y)

# Spliting the data into training set & test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

