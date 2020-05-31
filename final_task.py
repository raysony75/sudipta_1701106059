#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
column = ['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']
for i in column :
   data = pd.read_csv(r'C:\Users\hp\Downloads\spectrum _1\final\DS_ML_FinalTask\aa\student math_2.csv')
   b = data[i].unique()
   print(b)
   label_encoder = preprocessing.LabelEncoder()
   data[i] = label_encoder.fit_transform(data[i])
   a = data[i].unique()
   print(a)



# In[14]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv(r'C:\Users\hp\Downloads\spectrum _1\final\DS_ML_FinalTask\aa\student math_2.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
X[:,1] = labelencoder_X.fit_transform(X[:,1])
print(X)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print(y)


# In[8]:


import pandas as pd
data = pd.read_csv(r'C:\Users\hp\Downloads\spectrum _1\final\DS_ML_FinalTask\aa\student math_2.csv')
data.tail(n=400)
a = data[['G1','G2','G3']].tail(400)
print(a)
b = a.sum(axis=1)
print(b)
loc = 33
column = ('final grade')
value = b
data.head(400)
data.insert(loc,column,value,allow_duplicates = False)
data.head(400)


# In[6]:


import pandas as pd
import numpy as np 
data = pd.read_csv(r'C:\Users\hp\Downloads\spectrum _1\final\DS_ML_FinalTask\aa\student math_2.csv')
data.tail(n=400)
a = data[['G1','G2','G3']].tail(400)
print(a)
b = a.sum(axis=1)
print(b)
loc = 33
column = ('final grade')
value = b
data.head(400)
data.insert(loc,column,value,allow_duplicates = False)
data.head(400)
data.dropna(inplace = True)
y = pd.DataFrame(data['final grade'])
print(y.to_numpy())


# In[7]:


import pandas as pd
data = pd.read_csv(r'C:\Users\hp\Downloads\spectrum _1\final\DS_ML_FinalTask\aa\student math_2.csv')
data.tail(n=400)
a = data[['G1','G2','G3']].tail(400)
print(a)
b = a.sum(axis=1)
print(b)
loc = 33
column = ('final grade')
value = b
data.head(400)
data.insert(loc,column,value,allow_duplicates = False)
data.head(400)
x_cols = [x for x in data.columns if x != 'G3']
x_2 = data[x_cols]
print(x_2)


# In[13]:


import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv(r'C:\Users\hp\Downloads\spectrum _1\final\DS_ML_FinalTask\aa\student math_2.csv')
data.head(400)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)
print('X_train:\n')
print(X_train)
print(X_train.shape)
print('X_test:\n')
print(X_test)
print(X_test.shape)
print('y_train:\n')
print(y_train)
print(y_train.shape)
print('y_test:\n')
print(y_test)
print(y_test.shape)


# Final Task_2

# In[21]:


#import all neccesary packages
import numpy as np
import pandas as pd
from sklearn import utils
import matplotlib.pyplot as plt
import pdb;
pdb.set_trace()

#import the dataframe
df = pd.read_csv(r'C:\Users\hp\Downloads\spectrum _1\final\DS_ML_FinalTask\aa\student math_2.csv',sep = ";")

#Encoding the nominal values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df['school']= le.fit_transform(df['school'])
df['Pstatus']= le.fit_transform(df['Pstatus'])
df['sex'] = le.fit_transform(df['sex'])
df['address'] = le.fit_transform(df['address'])
df['famsize']= le.fit_transform(df['famsize'])
df['Mjob']= le.fit_transform(df['Mjob'])
df['Fjob']= le.fit_transform(df['Fjob'])
df['reason']= le.fit_transform(df['reason'])
df['guardian']= le.fit_transform(df['guardian'])
df['schoolsup']= le.fit_transform(df['schoolsup'])
df['famsup']= le.fit_transform(df['famsup'])
df['paid']= le.fit_transform(df['paid'])
df['activities']= le.fit_transform(df['activities'])
df['nursery']= le.fit_transform(df['nursery'])
df['higher']= le.fit_transform(df['higher'])
df['internet']= le.fit_transform(df['internet'])
df['romantic']= le.fit_transform(df['romantic'])

#creating a new column final_grade in the dataframe
col = df.loc[: , "G1":"G3"]
df['final_grade'] = col.mean(axis=1)

#new csv file after final_grade
df.to_csv(r'C:\Users\hp\Downloads\spectrum _1\final\DS_ML_FinalTask\aa\student math_2.csv',sep = ";")

#store final_grade column as an array in y
y = df['final_grade'].to_numpy()

#store all columns upto G2 in x as array
col = df.loc[:,"reason":"G2"]
x = col.to_numpy()

#function to predict and test the accuracy of the predicted value with the true value
def predict(x):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y)

    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(x_train,y_train)
    
    prediction = reg.predict(x_test)
    
    #calculate the accuracy
    from sklearn.metrics import r2_score
    accuracy = r2_score(y_test, prediction)
    print("Accuracy = ",accuracy*100, "%")
    
    #plot scatter plot between true value and predicted value
    plt.scatter(y_test, prediction, color = 'b')
    plt.xlabel('True Value --->')
    plt.ylabel('Predicted Value --->')
    plt.show()

#Backward Elimination to get highest accuracy out of predicted values
import statsmodels.api as sm
def bkwdelm(x,sl):
    numvars = len(x[0])
    for i in range(0,numvars):
        reg_OLS = sm.OLS(y,x).fit()
        maxvar = max(reg_OLS.pvalues).astype(float)
        if maxvar > sl:
           for j in range(0,numvars-i):
               if (reg_OLS.pvalues[j].astype(float) == maxvar):
                  x = np.delete(x,j,1)
    print(reg_OLS.summary())
    return x
    
sl = 0.005
x_opt = x[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]
x_modeled = bkwdelm(x_opt, sl)

predict(x_modeled)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




