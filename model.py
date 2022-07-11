from copyreg import pickle
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import pickle

data_filename = "./crime-data-from-2020-to-present/Crime_Data_from_2010_to_2019.csv"
df = pd.read_csv(data_filename)
df.columns = df.columns.str.replace(' ','_')
df.DATE_OCC = pd.to_datetime(df.DATE_OCC)
to_drop =["Weapon_Used_Cd",
          "Weapon_Desc",
          "Crm_Cd_2",
          "Crm_Cd_3",
          "Crm_Cd_4",
          "Crm_Cd_Desc",
         "Cross_Street"]
df.drop(to_drop, inplace=True, axis=1)
to_drop =["Rpt_Dist_No",
          "Crm_Cd_1",
          "Premis_Cd",
          "Crm_Cd",
          "Part_1-2",
         "Date_Rptd",
         "AREA_NAME",
         "Mocodes",
         "Premis_Desc",
         "Status"]
df.drop(to_drop, inplace=True, axis=1)
to_drop =["LOCATION",
         "Status_Desc",
         "DATE_OCC"]
df.drop(to_drop, inplace=True, axis=1)
df["LAT"].replace({0: np.nan}, inplace=True)  
df["Vict_Age"].replace({0: np.nan}, inplace=True)   
mean_value_age=df['Vict_Age'].mean()
df['Vict_Age'].fillna(value=mean_value_age, inplace=True)
sex = df["Vict_Sex"]
new_sex = list()
for i in sex:
    if i == 'M':
        new_sex.append(1)
    elif i == 'F':
        new_sex.append(0)
df['Vict_Sex'] = pd.Series(new_sex)
df.head()
mean_value_sex=df['Vict_Sex'].mean()
df['Vict_Sex'].fillna(value=mean_value_sex, inplace=True)
des = df["Vict_Descent"]
new_des = list()
for i in des:
    if i == 'H':
        new_des.append(1)
    elif i == "W":
        new_des.append(2)
    elif i == 'O':
        new_des.append(0)
df['Vict_Descent'] = pd.Series(new_des)
mean_value_des=df['Vict_Descent'].mean()
df['Vict_Descent'].fillna(value=mean_value_des, inplace=True)
df['Vict_Sex'] = df['Vict_Sex'].astype(int)
df['Vict_Descent'] = df['Vict_Descent'].astype(int)
df = df.dropna()
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = df.drop('Vict_Sex' ,axis=1)
Y = df['Vict_Sex']
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.1)
xtrain
from sklearn.tree import DecisionTreeClassifier
ly = DecisionTreeClassifier()
ly.fit(xtrain, ytrain)
pickle.dump(ly, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))


