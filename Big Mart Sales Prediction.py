# import required modules
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score


# loading the dataset
Sales_dataset = pd.read_csv("Train.csv")
# show the dataset
Sales_dataset.head()
Sales_dataset.tail()
# show the dataset shape
Sales_dataset.shape
# show some statistical info about the dataset
Sales_dataset.describe()


# check if there is any none values in the dataset to decide if will make a data cleaning or not
Sales_dataset.isnull().sum()
# make a data cleaning
datatypes = Sales_dataset.dtypes
print(datatypes)
for IndexDT in range(len(datatypes)):
    if datatypes[IndexDT]!="object":
        Sales_dataset.iloc[:,IndexDT].fillna(Sales_dataset.iloc[:,IndexDT].mean(),inplace=True)
    else:
        Sales_dataset.iloc[:,IndexDT].fillna(Sales_dataset.iloc[:,IndexDT].mode()[0],inplace=True)



# plot the distribution of Item_Outlet_Sales
plt.figure(figsize=(5,5))
sns.distplot(Sales_dataset['Item_Outlet_Sales'],color='blue')
# plot the distribution of Item_MRP
plt.figure(figsize=(5,5))
sns.distplot(Sales_dataset['Item_MRP'],color='red')
# plot the distribution of Item_Weight
plt.figure(figsize=(5,5))
sns.distplot(Sales_dataset['Item_Weight'],color='green')
# plot the distribution of Item_Visibility
plt.figure(figsize=(5,5))
sns.distplot(Sales_dataset['Item_Visibility'],color='yellow')

# find the countplot of the Item_Type column
plt.figure(figsize=(30,6))
Sales_dataset['Item_Type'].value_counts()
sns.countplot(x = 'Item_Type',data=Sales_dataset)
plt.show()

plt.figure(figsize=(10,10))
# find the groups of the Outlet_Location_Type column and count its repetition
sns.catplot(x = 'Outlet_Location_Type',data=Sales_dataset,kind='count')
# find the groups of the Outlet_Establishment_Year column and count its repetition
sns.catplot(x = 'Outlet_Establishment_Year',data=Sales_dataset,kind='count')
# find the groups of the Item_Fat_Content column and count its repetition
sns.catplot(x = 'Item_Fat_Content',data=Sales_dataset,kind='count')




# check of the values of the Item_Fat_Content column and its repetition
Sales_dataset['Item_Fat_Content'].value_counts()
# from the values of the column deduced that the data not cleared or cleaned
# so will clean the data by replacing LF, low fat values with Low Fat and reg value with Regular  
Sales_dataset.replace({'Item_Fat_Content':{'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'}},inplace=True)        
# check of the values of the Item_Fat_Content column and its repetition
Sales_dataset['Item_Fat_Content'].value_counts()




Sales_dataset.head()
# create a label encoder
le = LabelEncoder()
# convert the textual columns into numeric columns
datatypes = Sales_dataset.dtypes
print(datatypes)
for IndexDT in range(len(datatypes)):
    if datatypes[IndexDT]=="object":
        Sales_dataset.iloc[:,IndexDT] = le.fit_transform(Sales_dataset.iloc[:,IndexDT])
Sales_dataset.head()





# Split dataset into input and label data
X = Sales_dataset.drop(columns=['Item_Outlet_Sales'],axis=1)
Y = Sales_dataset['Item_Outlet_Sales']
print(X)
print(Y)
# Split the data into train and test data
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.6,random_state=8)
print(X.shape,x_train.shape,x_test.shape)
print(Y.shape,y_train.shape,y_test.shape)
# create the model and train it
XGBRModel = XGBRegressor()
XGBRModel.fit(x_train,y_train)
# make the model predict the train and test input data
predicted_train_data = XGBRModel.predict(x_train)
predicted_test_data = XGBRModel.predict(x_test)
# avaluate the model with r2 score 
accuracy_train_value = r2_score(predicted_train_data,y_train)
accuracy_test_value = r2_score(predicted_test_data,y_test)
print(accuracy_train_value,accuracy_test_value)
