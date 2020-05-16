#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
data = pd.read_csv(r'C:\Users\hp\Downloads\spectrum _1\DS_ML_Task1\f\student math_1.csv')
data.head(n=400)


# In[32]:


import pandas as pd
data = pd.read_csv(r'C:\Users\hp\Downloads\spectrum _1\DS_ML_Task1\f\student math_1.csv')
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


# In[33]:


import pandas as pd
data = pd.read_csv(r'C:\Users\hp\Downloads\spectrum _1\DS_ML_Task1\f\student math_1.csv')
data.drop(['G1','G2','G3'],axis = 1, inplace = True)
data.head(n=400)


# In[39]:


import pandas as pd
from pandas import DataFrame
data = pd.read_csv(r'C:\Users\hp\Downloads\spectrum _1\DS_ML_Task1\f\student math_1.csv')
a = DataFrame(data,columns = ['schoolsup'])
print(a)
data.loc[data['schoolsup']== 'yes','schoolsup'] = 1
data.loc[data['schoolsup']== 'no','schoolsup'] = 0
a = data[['schoolsup']].tail(400)
print(a)

b = DataFrame(data,columns = ['famsup'])
print(b)
data.loc[data['famsup']== 'yes','famsup'] = 1
data.loc[data['famsup']== 'no','famsup'] = 0
b = data[['famsup']].tail(400)
print(b)

c = DataFrame(data,columns = ['paid'])
print(c)
data.loc[data['paid']== 'yes','paid'] = 1
data.loc[data['paid']== 'no','paid'] = 0
c = data[['paid']].tail(400)
print(c)

d = DataFrame(data,columns = ['activities'])
print(d)
data.loc[data['activities']== 'yes','activities'] = 1
data.loc[data['activities']== 'no','activities'] = 0
d = data[['activities']].tail(400)
print(d)

e = DataFrame(data,columns = ['nursery'])
print(e)
data.loc[data['nursery']== 'yes','nursery'] = 1
data.loc[data['nursery']== 'no','nursery'] = 0
e = data[['nursery']].tail(400)
print(e)

f = DataFrame(data,columns = ['higher'])
print(f)
data.loc[data['higher']== 'yes','higher'] = 1
data.loc[data['higher']== 'no','higher'] = 0
f = data[['higher']].tail(400)
print(f)

g = DataFrame(data,columns = ['internet'])
print(g)
data.loc[data['internet']== 'yes','internet'] = 1
data.loc[data['internet']== 'no','internet'] = 0
g = data[['internet']].tail(400)
print(g)

h = DataFrame(data,columns = ['romantic'])
print(h)
data.loc[data['schoolsup']== 'yes','romantic'] = 1
data.loc[data['romantic']== 'no','romantic'] = 0
h = data[['romantic']].tail(400)
print(h)



# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv(r'C:\Users\hp\Downloads\spectrum _1\DS_ML_Task1\f\student math_1.csv')
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
d = data.plot(kind = 'scatter',y ='studytime',x = 'final grade',c = 'black',colormap ='viridis')
e = data.plot(kind = 'density',y ='studytime',x = 'final grade',c = 'red')
f = data.plot(y ='studytime',x = 'final grade',c = 'blue')


# In[30]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv(r'C:\Users\hp\Downloads\spectrum _1\DS_ML_Task1\f\student math_1.csv')
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
d = data.plot(kind = 'scatter',y ='studytime',x = 'final grade',c = 'black',colormap ='viridis')
d = data.boxplot(by = 'final grade' , column = ['studytime'],grid = False)


# In[ ]:





# In[ ]:




