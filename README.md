# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
  import pandas as pd
  import numpy as np
  import seaborn as sns
  
  from sklearn.model_selection import train_test_split
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.metrics import accuracy_score, confusion_matrix
  
  data=pd.read_csv("income(1) (1).csv",na_values=[ " ?"])
  data
```
![image](https://github.com/user-attachments/assets/d8b0bc0e-ed7c-4605-a55c-6c4705de4f90)

```
  data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/ab48899e-6c22-4354-bd69-bdfbe6d580c0)

```
  missing=data[data.isnull().any(axis=1)]
  missing  
```
![image](https://github.com/user-attachments/assets/9ceb7738-8df0-4d39-8a3d-1f38ce7437d8)

```
 data2=data.dropna(axis=0)
 data2
```
![image](https://github.com/user-attachments/assets/f5a4eb0c-b120-41fa-a39a-6516ac38a4be)

```
 sal=data["SalStat"]
 data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
 print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/cd24e806-4711-44cc-8962-6452c2da7b26)

```
 sal2=data2['SalStat']
 dfs=pd.concat([sal,sal2],axis=1)
 dfs
```
 ![image](https://github.com/user-attachments/assets/2bbeee21-8211-4e09-9da6-d7ab1d44aa64)

```
 data2
```
![image](https://github.com/user-attachments/assets/0d2ab41d-d18b-4be0-acb2-c33d7ae4f9d6)

```
 new_data=pd.get_dummies(data2, drop_first=True)
 new_data
```
![image](https://github.com/user-attachments/assets/695b14fd-89d8-4d20-8363-49ce55e8eeeb)

```
 columns_list=list(new_data.columns)
 print(columns_list)
```
![image](https://github.com/user-attachments/assets/af14b3ef-62d3-46dc-bdbc-653a126beb0b)

```
 features=list(set(columns_list)-set(['SalStat']))
 print(features)
```
![image](https://github.com/user-attachments/assets/2d11090e-890a-4eaa-badb-a126abe98ab3)

```
 y=new_data['SalStat'].values
 print(y)
```
![image](https://github.com/user-attachments/assets/2fab5920-cfb8-44d8-acd7-7625faf08e3c)

```
 x=new_data[features].values
 print(x)
```
![image](https://github.com/user-attachments/assets/69979f4a-ca40-4f40-95e6-b3d29edcc208)

```
 train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
 KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
 KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/23063a03-1bb4-4d7d-8477-cbe9306a3eec)

```
 prediction=KNN_classifier.predict(test_x)
 confusionMatrix=confusion_matrix(test_y, prediction)
 print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/8836b69b-1fbb-4001-bba4-6df61f07f866)

```
 accuracy_score=accuracy_score(test_y,prediction)
 print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/ef262df5-238c-444a-9aac-fdcc41df68fe)

```
 print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/user-attachments/assets/78c044c0-3e14-4d02-ba18-cb56bcc87900)

```
 data.shape
```
![image](https://github.com/user-attachments/assets/7bbbce89-c60b-4b0d-85d1-841a0de31a78)

```
 import pandas as pd
 import numpy as np
 from scipy.stats import chi2_contingency 
 import seaborn as sns
 tips=sns.load_dataset('tips')
 tips.head()
```
![image](https://github.com/user-attachments/assets/630ef309-4f12-46ef-a572-419ddef1d77c)

```
 tips.time.unique()
```
![image](https://github.com/user-attachments/assets/247a40c8-4b99-43ec-942c-2fae82521a97)

```
 contingency_table=pd.crosstab(tips['sex'],tips['time'])
 print(contingency_table)
```
![image](https://github.com/user-attachments/assets/da837f2c-0500-4ea2-aacb-b770793e1bb5)

```
 chi2,p,_,_=chi2_contingency(contingency_table)
 print(f"Chi-Square Statistics: {chi2}")
 print(f"P-Value: {p}")
```
![image](https://github.com/user-attachments/assets/55e1262e-3d03-4722-b0f2-30b6d0c6070e)

# RESULT:
  Thus, Feature selection and Feature scaling has been used on thegiven dataset.
