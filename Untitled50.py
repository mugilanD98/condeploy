import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


data=pd.read_excel("D:/mugimainpro/Concrete_Data.xls")


# In[ ]:


data1=data.rename(columns={'Cement (component 1)(kg in a m^3 mixture)':'Cement','Blast Furnace Slag (component 2)(kg in a m^3 mixture)':'Blast Furnace Slag','Fly Ash (component 3)(kg in a m^3 mixture)':'Fly Ash','Water  (component 4)(kg in a m^3 mixture)':'Water','Superplasticizer (component 5)(kg in a m^3 mixture)':'Superplasticizer','Coarse Aggregate  (component 6)(kg in a m^3 mixture)':'Coarse Aggregate','Fine Aggregate (component 7)(kg in a m^3 mixture)':'Fine Aggregate','Age (day)':'age','Concrete compressive strength(MPa, megapascals) ':'Concrete compressive strength'},inplace=False)


# In[ ]:


q0,q1,q2,q3,q4=np.percentile(data1['Blast Furnace Slag'],[0,25,50,75,100])
IQR=q3-q1
print("0th percentile : ",q0)
print("25th percentile : ",q1)
print("50th percentile : ",q2)
print("75th percentile : ",q3)
print("100th percentile : ",q4)
print("Inter quartile range : ",IQR)

upper=q3+(1.5*IQR)
lower=q1-(1.5*IQR)
data1[(data1['Blast Furnace Slag']<lower)| (data1['Blast Furnace Slag']>upper)]
data1.drop(data1[(data1['Blast Furnace Slag']<lower)|(data1['Blast Furnace Slag']>upper)].index,inplace=True)


# In[ ]:


q0,q1,q2,q3,q4=np.percentile(data1['Superplasticizer'],[0,25,50,75,100])
IQR=q3-q1
print("0th percentile : ",q0)
print("25th percentile : ",q1)
print("50th percentile : ",q2)
print("75th percentile : ",q3)
print("100th percentile : ",q4)
print("Inter quartile range : ",IQR)

upper=q3+(1.5*IQR)
lower=q1-(1.5*IQR)
data1[(data1['Superplasticizer']<lower)| (data1['Superplasticizer']>upper)]
data1.drop(data1[(data1['Superplasticizer']<lower)|(data1['Superplasticizer']>upper)].index,inplace=True)


# In[ ]:


q0,q1,q2,q3,q4=np.percentile(data1['Fine Aggregate'],[0,25,50,75,100])
IQR=q3-q1
print("0th percentile : ",q0)
print("25th percentile : ",q1)
print("50th percentile : ",q2)
print("75th percentile : ",q3)
print("100th percentile : ",q4)
print("Inter quartile range : ",IQR)

upper=q3+(1.5*IQR)
lower=q1-(1.5*IQR)
data1[(data1['Fine Aggregate']<lower)| (data1['Fine Aggregate']>upper)]
data1.drop(data1[(data1['Fine Aggregate']<lower)|(data1['Fine Aggregate']>upper)].index,inplace=True)


# In[ ]:


q0,q1,q2,q3,q4=np.percentile(data1['age'],[0,25,50,75,100])
IQR=q3-q1
print("0th percentile : ",q0)
print("25th percentile : ",q1)
print("50th percentile : ",q2)
print("75th percentile : ",q3)
print("100th percentile : ",q4)
print("Inter quartile range : ",IQR)

upper=q3+(1.5*IQR)
lower=q1-(1.5*IQR)
data1[(data1['age']<lower)| (data1['age']>upper)]
data1.drop(data1[(data1['age']<lower)|(data1['age']>upper)].index,inplace=True)


# In[ ]:


q0,q1,q2,q3,q4=np.percentile(data1['age'],[0,25,50,75,100])
IQR=q3-q1
print("0th percentile : ",q0)
print("25th percentile : ",q1)
print("50th percentile : ",q2)
print("75th percentile : ",q3)
print("100th percentile : ",q4)
print("Inter quartile range : ",IQR)

upper=q3+(1.5*IQR)
lower=q1-(1.5*IQR)
data1[(data1['age']<lower)| (data1['age']>upper)]
data1.drop(data1[(data1['age']<lower)|(data1['age']>upper)].index,inplace=True)


# In[ ]:


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[ ]:


x111=data1[['Cement','Blast Furnace Slag','Fly Ash','Superplasticizer','age']]
y111=data1['Concrete compressive strength']
x11_train,x11_test,y11_train,y11_test=train_test_split(x111,y111,test_size=0.2,random_state=24)
rf_reg1=RandomForestRegressor(n_estimators=50,random_state=0)
rf_reg1.fit(x11_train,y11_train)
r_pred1=rf_reg1.predict(x11_test)
mae=metrics.mean_absolute_error(y11_test,r_pred1)
mse=metrics.mean_squared_error(y11_test,r_pred1)
r2=metrics.r2_score(y11_test,r_pred1)
print(mae,mse,r2)


# In[ ]:


import pickle
filename="saved_model111"
pickle.dump(rf_reg1,open(filename,'wb'))

