#!/usr/bin/env python
# coding: utf-8

# In[69]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statistics
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA


# In[3]:


data = pd.read_csv('MELBOURNE_HOUSE_PRICES_LESS.csv')


# In[4]:


dh = data.head(11)
dh


# In[5]:


data.describe()


# In[6]:


data.info()


# In[7]:


data.isnull().sum()


# In[8]:


df = data.dropna(axis=0, subset=['Price'])
df


# In[9]:


df.isnull().sum()


# In[10]:


df.drop(['Address','Date'],axis=1,inplace=True)
df


# In[11]:


df.shape


# # Analysis Price
# 

# In[18]:


sns.displot(df['Price'])


# In[21]:


print("Skewness: %f" % data['Price'].skew())


# In[28]:


corr = df.corr()
sns.heatmap(corr, vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(5, 220, n=200),square=True);


# In[30]:


df['Rooms'].value_counts()


# In[40]:


data.Price.groupby(data['Rooms']).mean()


# In[41]:


sns.boxplot(x="Rooms", y="Price", data=data)


# In[43]:


data.Price.groupby(data['Type']).mean()


# In[48]:


sns.distplot(data[data['Type']=='h'].Price, hist = True, kde = True)
#orange
sns.distplot(data[data['Type']=='u'].Price, hist = True, kde = True)
#blue
sns.distplot(data[data['Type']=='t'].Price, hist = True, kde = True)
#green


# In[101]:


df1=df[['Type', 'Method', 'Regionname','Propertycount', 'Distance', 'Postcode','Rooms', 'Price']]


# In[102]:


X = df1.iloc[:,:-1]
Y = df1.iloc[:,-1]
data.shape


# In[103]:


X['Rooms'] = X['Rooms'].astype('object')


# ### holdout

# In[104]:


X = pd.get_dummies(X)


# ### split data

# In[105]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2 ,random_state=1)


# In[106]:


scaler = MinMaxScaler()
X_train.iloc[:,:4] = scaler.fit_transform(X_train.iloc[:, :4])
X_test.iloc[:,:4] = scaler.fit_transform(X_test.iloc[:, :4])


# In[112]:


regressor=LinearRegression()
regressor.fit(X_train,Y_train)


# In[113]:


Y_pred_hd = regressor.predict(X_test)
Y_pred_hd = Y_pred_hd.astype(int)


# In[115]:


test_result = np.concatenate((Y_pred_hd.reshape(len(Y_pred_hd),1),Y_test.values.reshape(len(Y_test),1)), axis=1)
test_result = pd.DataFrame(data = test_result, columns =['Y_Predict','Y_test'] )
test_result


# In[117]:


sns.scatterplot(x="Y_Predict", y="Y_test", data=test_result)


# In[118]:


print('Mean Square Error: {}'.format(mean_squared_error(Y_test, Y_pred_hd)))
print('R2 Score: {}'.format(r2_score(Y_test, Y_pred_hd)))
print('Root Mean Square Error: {}'.format(np.sqrt(mean_squared_error(Y_test, Y_pred_hd))))
print('Percentage between RMSE and Y-Mean: {}%'.format(int((np.sqrt(mean_squared_error(Y_test, Y_pred_hd)))/(Y.mean())*100)))


# In[124]:


from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(regressor, hist=False)
visualizer.fit(X_train, Y_train) 
visualizer.score(X_test, Y_test) 
visualizer.show() 


# ## cross validation

# In[126]:


k_fold = []
val_score = []
for i in range(2,11):
    scores = cross_val_score(LinearRegression(), X, Y, cv=i)
    print('Cross validation score with {} fold: {}'.format(i, scores.mean()))
    k_fold.append(i)
    val_score.append(scores.mean())


# In[131]:


Y_pred_cv = cross_val_predict(regressor, X, Y, cv=3)
Y_pred_cv = Y_pred_cv.astype(int)


# In[132]:


test_result_cv = np.concatenate((Y_pred_cv.reshape(len(Y_pred_cv),1),Y.values.reshape(len(Y),1)), axis=1)
test_result_cv = pd.DataFrame(data = test_result_cv, columns =['Y_predict','Y'] )
test_result_cv


# In[133]:


sns.scatterplot(x="Y_predict", y="Y", data=test_result_cv)


# In[141]:


print('Mean Square Error: {}'.format(mean_squared_error(Y, Y_pred_cv)))
print('Root Mean Square Error: {}'.format(np.sqrt(mean_squared_error(Y, Y_pred_cv))))
print('Percentage between RMSE and Y-Mean: {}%'.format(int((np.sqrt(mean_squared_error(Y, Y_pred_cv)))/(Y.mean())*100)))


# # PCA

# In[125]:


pca = PCA()
pca.fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_*100))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# In[135]:


PCA(n_components = 20)
X_train_pca = PCA(n_components = 20).fit_transform(X_train)
X_test_pca= PCA(n_components = 20).fit_transform(X_test)


# In[136]:


regressor_pca = LinearRegression()
regressor_pca.fit(X_train_pca,Y_train)


# In[137]:


Y_pred_hd_pca = regressor_pca.predict(X_test_pca)
Y_pred_hd_pca = Y_pred_hd_pca.astype(int)
test_result_pca = np.concatenate((Y_pred_hd_pca.reshape(len(Y_pred_hd_pca),1),Y_test.values.reshape(len(Y_test),1)), axis=1)
test_result_pca = pd.DataFrame(data = test_result_pca, columns =['Y_Predict','Y_test'] )
test_result_pca


# In[138]:


sns.scatterplot(x="Y_Predict", y="Y_test", data=test_result_pca)


# In[140]:


print('Mean Square Error: {}'.format(mean_squared_error(Y_test, Y_pred_hd_pca)))
print('Root Mean Square Error: {}'.format(np.sqrt(mean_squared_error(Y_test, Y_pred_hd_pca))))
print('Percentage between RMSE and Y-Mean: {}%'.format(int((np.sqrt(mean_squared_error(Y_test, Y_pred_hd_pca)))/(Y.mean())*100)))


# # PCA cross validation

# # conclude
# ## - Both module holdout and cross validation are incorrectly to predict data 
# ## - Maybe there have some effective feature that i ignore to make it more correctly
