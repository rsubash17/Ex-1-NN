<H3>ENTER YOUR NAME       : Subash R</H3>
<H3>ENTER YOUR REGISTER NO: 212223230218</H3>
<H3>EX.NO:1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```py
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))


```
## OUTPUT:
### Dataset:
<img width="1590" height="450" alt="image" src="https://github.com/user-attachments/assets/92cd51c7-9422-4a86-bb4f-040556c0ef75" />

### X Values:
<img width="473" height="295" alt="image" src="https://github.com/user-attachments/assets/06f812c7-8d0e-421f-91a1-6ca46539ea29" />

### Y Values:
<img width="302" height="141" alt="image" src="https://github.com/user-attachments/assets/a7851d8d-ae85-4a5c-8aa6-a2a6cfbabf3c" />

### Null Values:
<img width="231" height="382" alt="image" src="https://github.com/user-attachments/assets/301c0bba-5ddc-4daf-aa76-cfca8e90ccc1" />

### Duplicated Values:
<img width="267" height="312" alt="image" src="https://github.com/user-attachments/assets/07e00ab2-1db8-40d6-89e0-4428f8cc5363" />

### Description:
<img width="1557" height="310" alt="image" src="https://github.com/user-attachments/assets/27788572-6e6e-4a77-a7dd-253e356e0dcd" />

### Normalized Dataset:
<img width="827" height="666" alt="image" src="https://github.com/user-attachments/assets/f5a07ae6-11d8-4bc5-a585-b004b86794bd" />

### Training Data:
<img width="725" height="213" alt="image" src="https://github.com/user-attachments/assets/78567ac8-e5e5-4678-9c0c-449a8dd2a626" />

### Testing Data:
<img width="726" height="202" alt="image" src="https://github.com/user-attachments/assets/1c069e19-36bd-4113-b520-d7ca771e5c58" />



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


