#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction
# 
# 

# In[1]:


#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[2]:


#load data set
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


# In[3]:


print("shape of train:",train.shape)
print("shape of test:",test.shape)


# In[4]:


train.head(10)


# In[5]:


test.head(10)


# In[6]:


#concat train and test
df=pd.concat((train,test))
temp_df=df
print("shape of df:",df.shape)


# In[7]:


df.head(6)


# In[8]:


df.tail(6)


# # Exploratory Data Analysis (EDA)

# In[9]:


# To show the all columns
pd.set_option("display.max_columns", 2000)
pd.set_option("display.max_rows", 85)


# In[10]:


df.head(6)


# In[11]:


df.tail(6)


# In[12]:


df.info()


# In[13]:


df.describe()


# In[14]:


df.select_dtypes(include=['int64', 'float64']).columns


# In[15]:


df.select_dtypes(include=['object']).columns


# In[16]:


# Set index as Id column
df = df.set_index("Id")


# In[17]:


df.head(6)


# In[18]:


# Show the null values using heatmap
plt.figure(figsize=(16,9))
sns.heatmap(df.isnull())


# In[19]:


# Get the percentages of null value
null_percent = df.isnull().sum()/df.shape[0]*100
null_percent


# In[20]:


col_for_drop = null_percent[null_percent > 20].keys() # if the null value % 20 or > 20 so need to drop it


# In[21]:


# drop columns
df = df.drop(col_for_drop, "columns")
df.shape


# In[22]:


# find the unique value count
for i in df.columns:
    print(i + "\t" + str(len(df[i].unique())))


# In[23]:


# find unique values of each column
for i in df.columns:
    print("Unique value of:>>> {} ({})\n{}\n".format(i, len(df[i].unique()), df[i].unique()))


# In[24]:


# Describe the target 
train["SalePrice"].describe()


# In[25]:


# Plot the distplot of target
plt.figure(figsize=(10,8))
bar = sns.distplot(train["SalePrice"])
bar.legend(["Skewness: {:.2f}".format(train['SalePrice'].skew())])


# In[26]:


# correlation heatmap
plt.figure(figsize=(25,25))
ax = sns.heatmap(train.corr(), cmap = "coolwarm", annot=True, linewidth=2)

# to fix the bug "first and last row cut in half of heatmap plot"
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# In[27]:


# correlation heatmap of higly correlated features with SalePrice
hig_corr = train.corr()
hig_corr_features = hig_corr.index[abs(hig_corr["SalePrice"]) >= 0.5]
hig_corr_features


# In[28]:


plt.figure(figsize=(10,8))
ax = sns.heatmap(train[hig_corr_features].corr(), cmap = "coolwarm", annot=True, linewidth=3)
# to fix the bug "first and last row cut in half of heatmap plot"
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# In[29]:


# Plot regplot to get the nature of highly correlated data
plt.figure(figsize=(16,9))
for i in range(len(hig_corr_features)):
    if i <= 9:
        plt.subplot(3,4,i+1)
        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
        sns.regplot(data=train, x = hig_corr_features[i], y = 'SalePrice')


# # Handling Missing Value

# In[30]:


missing_col = df.columns[df.isnull().any()]
missing_col


# #Handling missing value of Bsmt feature

# In[31]:


bsmt_col = ['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1',
       'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF', 'TotalBsmtSF']
bsmt_feat = df[bsmt_col]
bsmt_feat


# In[32]:


bsmt_feat.info()


# In[33]:


bsmt_feat.isnull().sum()


# In[34]:


bsmt_feat = bsmt_feat[bsmt_feat.isnull().any(axis=1)]
bsmt_feat


# In[35]:


bsmt_feat_all_nan = bsmt_feat[(bsmt_feat.isnull() | bsmt_feat.isin([0])).all(1)]
bsmt_feat_all_nan


# In[36]:


bsmt_feat_all_nan.shape


# In[37]:


qual = list(df.loc[:, df.dtypes == 'object'].columns.values)
qual


# In[38]:


# Fillinf the mising value in bsmt features
for i in bsmt_col:
    if i in qual:
        bsmt_feat_all_nan[i] = bsmt_feat_all_nan[i].replace(np.nan,'NA') # replace the NAN value by 'NA'
    else:
        bsmt_feat_all_nan[i] = bsmt_feat_all_nan[i].replace(np.nan,0) # replace the NAN value inplace of 0

bsmt_feat.update(bsmt_feat_all_nan) # update bsmt_feat df by bsmt_feat_all_nan
df.update(bsmt_feat_all_nan) # update df by bsmt_feat_all_nan

"""
>>> df = pd.DataFrame({'A': [1, 2, 3],
...                    'B': [400, 500, 600]})
>>> new_df = pd.DataFrame({'B': [4, 5, 6],
...                        'C': [7, 8, 9]})
>>> df.update(new_df)
>>> df
   A  B
0  1  4
1  2  5
2  3  6
"""


# In[39]:


bsmt_feat = bsmt_feat[bsmt_feat.isin([np.nan]).any(axis=1)]
bsmt_feat


# In[40]:


bsmt_feat.shape


# In[41]:


print(df['BsmtFinSF2'].max())
print(df['BsmtFinSF2'].min())


# In[42]:


pd.cut(range(0,1526),5) # create a bucket


# In[43]:


df_slice = df[(df['BsmtFinSF2'] >= 305) & (df['BsmtFinSF2'] <= 610)]
df_slice


# In[44]:


bsmt_feat.at[333,'BsmtFinType2'] = df_slice['BsmtFinType2'].mode()[0] # replace NAN value of BsmtFinType2 by mode of buet ((305.0, 610.0)


# In[45]:


bsmt_feat


# In[46]:


bsmt_feat['BsmtExposure'] = bsmt_feat['BsmtExposure'].replace(np.nan, df[df['BsmtQual'] =='Gd']['BsmtExposure'].mode()[0])


# In[47]:


bsmt_feat['BsmtCond'] = bsmt_feat['BsmtCond'].replace(np.nan, df['BsmtCond'].mode()[0])
bsmt_feat['BsmtQual'] = bsmt_feat['BsmtQual'].replace(np.nan, df['BsmtQual'].mode()[0])


# In[48]:


df.update(bsmt_feat)


# In[49]:


bsmt_feat.isnull().sum()


# # Handling missing value of Garage feature

# In[50]:


df.columns[df.isnull().any()]


# In[52]:


garage_col = ['GarageArea', 'GarageCars', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'GarageYrBlt',]
garage_feat = df[garage_col]
garage_feat = garage_feat[garage_feat.isnull().any(axis=1)]
garage_feat


# In[53]:


garage_feat.shape


# In[55]:


garage_feat_all_nan = garage_feat[(garage_feat.isnull() | garage_feat.isin([0])).all(1)]
garage_feat_all_nan.shape


# In[56]:


for i in garage_feat:
    if i in qual:
        garage_feat_all_nan[i] = garage_feat_all_nan[i].replace(np.nan, 'NA')
    else:
        garage_feat_all_nan[i] = garage_feat_all_nan[i].replace(np.nan, 0)
        
garage_feat.update(garage_feat_all_nan)
df.update(garage_feat_all_nan)


# In[57]:


garage_feat = garage_feat[garage_feat.isnull().any(axis=1)]
garage_feat


# In[58]:


for i in garage_col:
    garage_feat[i] = garage_feat[i].replace(np.nan, df[df['GarageType'] == 'Detchd'][i].mode()[0])


# In[59]:


garage_feat.isnull().any()


# In[60]:


df.update(garage_feat)


# # Handling missing value of remain feature

# In[61]:


df.columns[df.isnull().any()]


# In[62]:


df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
df['Functional'] = df['Functional'].fillna(df['Functional'].mode()[0])
df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
df['Utilities'] = df['Utilities'].fillna(df['Utilities'].mode()[0])
df['MasVnrType'] = df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])


# In[63]:


df.columns[df.isnull().any()]


# In[64]:


df[df['MasVnrArea'].isnull() == True]['MasVnrType'].unique()


# In[65]:


df.loc[(df['MasVnrType'] == 'None') & (df['MasVnrArea'].isnull() == True), 'MasVnrArea'] = 0


# In[66]:


df.isnull().sum()/df.shape[0] * 100


# # Handling missing value of LotFrontage feature

# In[67]:


lotconfig = ['Corner', 'Inside', 'CulDSac', 'FR2', 'FR3']
for i in lotconfig:
    df['LotFrontage'] = pd.np.where((df['LotFrontage'].isnull() == True) & (df['LotConfig'] == i) , df[df['LotConfig'] == i] ['LotFrontage'].mean(), df['LotFrontage'])


# In[68]:


df.isnull().sum()


# # Feature Transformation

# In[69]:


df.columns


# In[70]:


# converting columns in str which have categorical nature but in int64
feat_dtype_convert = ['MSSubClass', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
for i in feat_dtype_convert:
    df[i] = df[i].astype(str)


# In[71]:


df['MoSold'].unique() # MoSold = Month of sold


# In[72]:


# conver in month abbrevation
import calendar
df['MoSold'] = df['MoSold'].apply(lambda x : calendar.month_abbr[x])


# In[73]:


df['MoSold'].unique()


# In[74]:


quan = list(df.loc[:, df.dtypes != 'object'].columns.values)


# In[75]:


quan


# In[76]:


len(quan)


# In[ ]:


# obj_feat = list(df.loc[:, df.dtypes == 'object'].columns.values)
obj_feat


# # Conver categorical code into order

# In[78]:


from pandas.api.types import CategoricalDtype
df['BsmtCond'] = df['BsmtCond'].astype(CategoricalDtype(categories=['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes


# In[79]:


df['BsmtCond'].unique()


# In[80]:


df['BsmtExposure'] = df['BsmtExposure'].astype(CategoricalDtype(categories=['NA', 'Mn', 'Av', 'Gd'], ordered = True)).cat.codes


# In[81]:


df['BsmtExposure'].unique()


# In[82]:


df['BsmtFinType1'] = df['BsmtFinType1'].astype(CategoricalDtype(categories=['NA', 'Unf', 'LwQ', 'Rec', 'BLQ','ALQ', 'GLQ'], ordered = True)).cat.codes
df['BsmtFinType2'] = df['BsmtFinType2'].astype(CategoricalDtype(categories=['NA', 'Unf', 'LwQ', 'Rec', 'BLQ','ALQ', 'GLQ'], ordered = True)).cat.codes
df['BsmtQual'] = df['BsmtQual'].astype(CategoricalDtype(categories=['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['ExterQual'] = df['ExterQual'].astype(CategoricalDtype(categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['ExterCond'] = df['ExterCond'].astype(CategoricalDtype(categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['Functional'] = df['Functional'].astype(CategoricalDtype(categories=['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod','Min2','Min1', 'Typ'], ordered = True)).cat.codes
df['GarageCond'] = df['GarageCond'].astype(CategoricalDtype(categories=['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['GarageQual'] = df['GarageQual'].astype(CategoricalDtype(categories=['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['GarageFinish'] = df['GarageFinish'].astype(CategoricalDtype(categories=['NA', 'Unf', 'RFn', 'Fin'], ordered = True)).cat.codes
df['HeatingQC'] = df['HeatingQC'].astype(CategoricalDtype(categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['KitchenQual'] = df['KitchenQual'].astype(CategoricalDtype(categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['PavedDrive'] = df['PavedDrive'].astype(CategoricalDtype(categories=['N', 'P', 'Y'], ordered = True)).cat.codes
df['Utilities'] = df['Utilities'].astype(CategoricalDtype(categories=['ELO', 'NASeWa', 'NASeWr', 'AllPub'], ordered = True)).cat.codes


# In[83]:


df['Utilities'].unique()


# # Show skewness of feature with distplot

# In[84]:


skewed_features = ['1stFlrSF',
 '2ndFlrSF',
 '3SsnPorch',
 'BedroomAbvGr',
 'BsmtFinSF1',
 'BsmtFinSF2',
 'BsmtFullBath',
 'BsmtHalfBath',
 'BsmtUnfSF',
 'EnclosedPorch',
 'Fireplaces',
 'FullBath',
 'GarageArea',
 'GarageCars',
 'GrLivArea',
 'HalfBath',
 'KitchenAbvGr',
 'LotArea',
 'LotFrontage',
 'LowQualFinSF',
 'MasVnrArea',
 'MiscVal',
 'OpenPorchSF',
 'PoolArea',
 'ScreenPorch',
 'TotRmsAbvGrd',
 'TotalBsmtSF',
 'WoodDeckSF']


# In[85]:


quan == skewed_features


# In[86]:


plt.figure(figsize=(25,20))
for i in range(len(skewed_features)):
    if i <= 28:
        plt.subplot(7,4,i+1)
        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
        ax = sns.distplot(df[skewed_features[i]])
        ax.legend(["Skewness: {:.2f}".format(df[skewed_features[i]].skew())], fontsize = 'xx-large')


# In[87]:


df_back = df


# In[88]:


# decrease the skewnwnes of the data
for i in skewed_features:
    df[i] = np.log(df[i] + 1)


# In[89]:


plt.figure(figsize=(25,20))
for i in range(len(skewed_features)):
    if i <= 28:
        plt.subplot(7,4,i+1)
        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
        ax = sns.distplot(df[skewed_features[i]])
        ax.legend(["Skewness: {:.2f}".format(df[skewed_features[i]].skew())], fontsize = 'xx-large')


# In[93]:


SalePrice = np.log(train['SalePrice'] + 1)


# In[91]:


# get object feature to conver in numeric using dummy variable
obj_feat = list(df.loc[:,df.dtypes == 'object'].columns.values)
len(obj_feat)


# In[92]:


# dummy varaibale
dummy_drop = []
clean_df = df
for i in obj_feat:
    dummy_drop += [i + '_' + str(df[i].unique()[-1])]

df = pd.get_dummies(df, columns = obj_feat)
df = df.drop(dummy_drop, axis = 1)


# In[94]:


df.shape


# In[95]:


#sns.pairplot(df)


# In[96]:


# scaling dataset with robust scaler
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(df)
df = scaler.transform(df)


# # Machine Learning Model Building

# In[97]:


train_len = len(train)


# In[98]:


X_train = df[:train_len]
X_test = df[train_len:]
y_train = SalePrice

print(X_train.shape)
print(X_test.shape)
print(len(y_train))


# # Cross Validation

# In[99]:


from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, r2_score

def test_model(model, X_train=X_train, y_train=y_train):
    cv = KFold(n_splits = 3, shuffle=True, random_state = 45)
    r2 = make_scorer(r2_score)
    r2_val_score = cross_val_score(model, X_train, y_train, cv=cv, scoring = r2)
    score = [r2_val_score.mean()]
    return score


# # Linear Regression

# In[100]:


import sklearn.linear_model as linear_model
LR = linear_model.LinearRegression()
test_model(LR)


# In[101]:


# Cross validation
cross_validation = cross_val_score(estimator = LR, X = X_train, y = y_train, cv = 10)
print("Cross validation accuracy of LR model = ", cross_validation)
print("\nCross validation mean accuracy of LR model = ", cross_validation.mean())


# In[102]:


rdg = linear_model.Ridge()
test_model(rdg)


# In[103]:


lasso = linear_model.Lasso(alpha=1e-4)
test_model(lasso)


# # Fitting Polynomial Regression To The Dataset
# from sklearn.preprocessing import PolynomialFeatures poly_reg = PolynomialFeatures(degree = 2) X_poly = poly_reg.fit_transform(X_train) poly_reg.fit(X_poly, y_train) lin_reg_2 = LinearRegression()
# 
# #Lin_reg_2.Fit(X_poly, Y_train)
# #test_model(Lin_reg_2,X_poly)

# # SVM

# In[116]:


from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
test_model(svr_reg)


# # Decision Tree Regressor

# In[117]:


from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=21)
test_model(dt_reg)


# # Random Forest Regressor

# In[118]:


from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 1000, random_state=51)
test_model(rf_reg)


# # Bagging & Boosting

# In[119]:


from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
br_reg = BaggingRegressor(n_estimators=1000, random_state=51)
gbr_reg = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, loss='ls', random_state=51)


# In[120]:


test_model(br_reg)


# In[121]:


test_model(gbr_reg)


# # XGBoost

# # SVM Model Bulding

# In[ ]:


SVM Model Bulding


# In[127]:


svr_reg.fit(X_train,y_train)
y_pred = np.exp(svr_reg.predict(X_test)).round(2)


# In[128]:


y_pred


# In[129]:


submit_test1 = pd.concat([test['Id'],pd.DataFrame(y_pred)], axis=1)
submit_test1.columns=['Id', 'SalePrice']


# In[130]:


submit_test1


# In[131]:


submit_test1.to_csv('sample_submission.csv', index=False )


# # SVM Model Bulding Hyperparameter Tuning

# # Hyperparameter Tuning
# from sklearn.model_selection import RandomizedSearchCV, GridSearchCV params = {‘kernel’: [‘linear’, ‘rbf’, ‘sigmoid’], ‘gamma’: [1, 0.1, 0.01, 0.001, 0.0001], ‘C’: [0.1, 1, 10, 100, 1000], ‘epsilon’: [1, 0.2, 0.1, 0.01, 0.001, 0.0001]}

# rand_search = RandomizedSearchCV(svr_reg, param_distributions=params, n_jobs=-1, cv=11) rand_search.fit(X_train, y_train) rand_search.bestparams

# In[137]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
params = {'kernel': ['rbf'],
         'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
         'C': [0.1, 1, 10, 100, 1000],
         'epsilon': [1, 0.2, 0.1, 0.01, 0.001, 0.0001]}
rand_search = RandomizedSearchCV(svr_reg, param_distributions=params, n_jobs=-1, cv=11)
rand_search.fit(X_train, y_train)
rand_search.best_score_


# In[138]:


svr_reg= SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.01, gamma=0.0001,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
test_model(svr_reg)


# In[139]:


svr_reg.fit(X_train,y_train)
y_pred = np.exp(svr_reg.predict(X_test)).round(2)


# In[140]:


y_pred


# In[141]:


submit_test3 = pd.concat([test['Id'],pd.DataFrame(y_pred)], axis=1)
submit_test3.columns=['Id', 'SalePrice']


# In[142]:


submit_test3.to_csv('sample_submission.csv', index=False)
submit_test3


# Name Submitted Wait time Execution time Score sample_submission.csv 3 days ago 0 seconds 0 seconds 0.12612

# # Feature Engineering / Selection To Improve Accuracy

# In[149]:


# correlation Barplot
plt.figure(figsize=(9,16))
corr_feat_series = pd.Series.sort_values(train.corrwith(train.SalePrice))
sns.barplot(x=corr_feat_series, y=corr_feat_series.index, orient='h')


# In[150]:


df_back1 = df_back


# In[151]:


df_back1.to_csv('df_for_feature_engineering.csv', index=False)


# In[152]:


list(corr_feat_series.index)


# # House Prices: Advanced Regression Techniques

# # Feature Selection / Engineering

# # Import Libraries

# In[153]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[154]:


df = pd.read_csv('df_for_feature_engineering.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
df


# In[155]:


#df = df.set_index('Id')


# # Drop Feature

# In[156]:


df = df.drop(['YrSold',
 'LowQualFinSF',
 'MiscVal',
 'BsmtHalfBath',
 'BsmtFinSF2',
 '3SsnPorch',
 'MoSold'],axis=1)


# In[157]:


quan = list(df.loc[:,df.dtypes != 'object'].columns.values)
quan


# In[158]:


skewd_feat = ['1stFlrSF',
 '2ndFlrSF',
 'BedroomAbvGr',
 'BsmtFinSF1',
 'BsmtFullBath',
 'BsmtUnfSF',
 'EnclosedPorch',
 'Fireplaces',
 'FullBath',
 'GarageArea',
 'GarageCars',
 'GrLivArea',
 'HalfBath',
 'KitchenAbvGr',
 'LotArea',
 'LotFrontage',
 'MasVnrArea',
 'OpenPorchSF',
 'PoolArea',
 'ScreenPorch',
 'TotRmsAbvGrd',
 'TotalBsmtSF',
 'WoodDeckSF']
#  '3SsnPorch',  'BsmtFinSF2',  'BsmtHalfBath',  'LowQualFinSF', 'MiscVal'


# In[159]:


# Decrease the skewness of the data
for i in skewd_feat:
    df[i] = np.log(df[i] + 1)
    
SalePrice = np.log(train['SalePrice'] + 1)


# # Decrease The Skewnwnes Of The Data

# for i in skewed_features: df[i] = np.log(df[i] + 1)

# In[161]:


df


# In[162]:


obj_feat = list(df.loc[:, df.dtypes == 'object'].columns.values)
print(len(obj_feat))

obj_feat


# In[163]:


# dummy varaibale
dummy_drop = []
for i in obj_feat:
    dummy_drop += [i + '_' + str(df[i].unique()[-1])]

df = pd.get_dummies(df, columns = obj_feat)
df = df.drop(dummy_drop, axis = 1)


# In[164]:


df.shape


# In[165]:


# scaling dataset with robust scaler
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(df)
df = scaler.transform(df)


# # Model Bulding

# In[166]:


train_len = len(train)
X_train = df[:train_len]
X_test = df[train_len:]
y_train = SalePrice

print("Shape of X_train: ", len(X_train))
print("Shape of X_test: ", len(X_test))
print("Shape of y_train: ", len(y_train))


# # Cross Validation

# In[167]:


from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, r2_score

def test_model(model, X_train=X_train, y_train=y_train):
    cv = KFold(n_splits = 3, shuffle=True, random_state = 45)
    r2 = make_scorer(r2_score)
    r2_val_score = cross_val_score(model, X_train, y_train, cv=cv, scoring = r2)
    score = [r2_val_score.mean()]
    return score


# In[168]:


# first cross validation with df with log second without log


# # Linear Model

# In[170]:


import sklearn.linear_model as linear_model
LR = linear_model.LinearRegression()
test_model(LR)


# In[171]:


rdg = linear_model.Ridge()
test_model(rdg)


# In[172]:


lasso = linear_model.Lasso(alpha=1e-4)
test_model(lasso)


# # Support Vector Machine

# In[173]:


from sklearn.svm import SVR
svr = SVR(kernel='rbf')
test_model(svr)


# # Svm Hyper Parameter Tuning

# In[174]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
params = {'kernel': ['rbf'],
         'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
         'C': [0.1, 1, 10, 100, 1000],
         'epsilon': [1, 0.2, 0.1, 0.01, 0.001, 0.0001]}
rand_search = RandomizedSearchCV(svr_reg, param_distributions=params, n_jobs=-1, cv=11)
rand_search.fit(X_train, y_train)
rand_search.best_score_


# In[175]:


rand_search.best_estimator_


# In[176]:


svr_reg1=SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
test_model(svr_reg1)


# In[177]:


svr_reg= SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.01, gamma=0.0001,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
test_model(svr_reg)


# # Solution

# In[181]:


svr_reg.fit(X_train,y_train)
y_pred = np.exp(svr_reg.predict(X_test)).round(2)
submit_test = pd.concat([test['Id'],pd.DataFrame(y_pred)], axis=1)
submit_test.columns=['Id', 'SalePrice']
submit_test.to_csv('sample_submission.csv', index=False)
submit_test

"""
file: sample_submission-v1-fs
rank: 1444
Red AI Productionnovice tier
0.12278
4
3m
Your Best Entry 
You advanced 140 places on the leaderboard!

Your submission scored 0.12278, which is an improvement of your previous score of 0.12484. Great job!"""


# # Model Save

# In[182]:


import pickle

pickle.dump(svr_reg, open('model_house_price_prediction.csv', 'wb'))
model_house_price_prediction = pickle.load(open('model_house_price_prediction.csv', 'rb'))
model_house_price_prediction.predict(X_test)


# In[183]:


test_model(model_house_price_prediction)


# # SVM Accuracy = 90%

# # Machine Learning Model Building Never End Until And Unless App Not Stop
