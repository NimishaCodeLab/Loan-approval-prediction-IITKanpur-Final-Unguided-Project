#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# # Step-1 : Import the required libraries

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[4]:


from sklearn.metrics import ConfusionMatrixDisplay


# In[5]:


from sklearn.preprocessing import LabelEncoder,MinMaxScaler


# In[6]:


from sklearn.linear_model import  LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[7]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[145]:


from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import SGDRegressor
from xgboost import XGBClassifier


# # Step-2: Read the dataset

# In[8]:


df=pd.read_csv("SBAnational.csv")


# # Step-3: Data Pre-Processing
# #Data Pre-Processing
# #Understanding the dataset 
# #and it's features and pre-processing it according to the required dataframe for model training purposes.

# In[9]:


df.head()


# In[10]:


df.info()


# In[11]:


df.shape


# In[12]:


df.describe()


# In[13]:


df.isnull().sum()


# In[14]:


df.dropna(subset=['Name', 'City', 'State', 'BankState', 'NewExist','RevLineCr', 'LowDoc', 'DisbursementDate', 'MIS_Status'], inplace=True)


# In[15]:


df.isnull().sum()


# In[16]:


df.dtypes


# In[17]:


df[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']].head()


# In[142]:


#Here, we notice that the issue is because of the dollar symbol, because of which converts the int dtype column to object.

#So we remove the symbol and convert the dtype of these columns.


# In[19]:


df[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']] = df[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']].applymap(lambda x: x.strip().replace('$', '').replace(',', ''))


# In[20]:


#Here, in case of ApprovalFY, we have two dtypes: str and int

#We need to convert into int dtype, so we need to check which unique values in the column has str dtype


# In[21]:


df['ApprovalFY'].apply(type).value_counts()


# In[22]:


df.ApprovalFY.unique()


# In[23]:


#So, we notice that there are some rows which have the character 'A' along with the year.


# In[24]:


def clean_str(x):
    if isinstance(x, str):
        return x.replace('A', '')
    return x

df.ApprovalFY = df.ApprovalFY.apply(clean_str).astype('int64')


# In[25]:


df['ApprovalFY'].apply(type).value_counts()


# In[26]:


#Further converting the columns to their appropriate dtypes.


# In[27]:


df = df.astype({'Zip': 'str', 'NewExist': 'int64', 'UrbanRural': 'str', 'DisbursementGross': 'float', 'BalanceGross': 'float',
                          'ChgOffPrinGr': 'float', 'GrAppv': 'float', 'SBA_Appv': 'float'})


# In[28]:


df.dtypes


# In[29]:


df['Industry'] = df['NAICS'].astype('str').apply(lambda x: x[:2])


# In[30]:


#Here, according the file that was provided along with the dataset, we map the industry code with the industry name.


# In[31]:


df['Industry'] = df['Industry'].map({
    '11': 'Ag/For/Fish/Hunt',
    '21': 'Min/Quar/Oil_Gas_ext',
    '22': 'Utilities',
    '23': 'Construction',
    '31': 'Manufacturing',
    '32': 'Manufacturing',
    '33': 'Manufacturing',
    '42': 'Wholesale_trade',
    '44': 'Retail_trade',
    '45': 'Retail_trade',
    '48': 'Trans/Ware',
    '49': 'Trans/Ware',
    '51': 'Information',
    '52': 'Finance/Insurance',
    '53': 'RE/Rental/Lease',
    '54': 'Prof/Science/Tech',
    '55': 'Mgmt_comp',
    '56': 'Admin_sup/Waste_Mgmt_Rem',
    '61': 'Educational',
    '62': 'Healthcare/Social_assist',
    '71': 'Arts/Entertain/Rec',
    '72': 'Accom/Food_serv',
    '81': 'Other_no_pub',
    '92': 'Public_Admin'
})


# In[32]:


df.dropna(subset = ['Industry'], inplace = True)


# In[33]:


#Now, we convert the FranchiseCode column to a binary column, 
#based on if whether it is or it isn't a franchise, referring the file along with the dataset.


# In[34]:


df.FranchiseCode.unique()


# In[35]:


df.loc[(df['FranchiseCode'] <= 1), 'IsFranchise'] = 0
df.loc[(df['FranchiseCode'] > 1), 'IsFranchise'] = 1


# In[36]:


df.FranchiseCode


# In[37]:


#Convert the NewExist column into a binary column.


# In[38]:


df = df[(df['NewExist'] == 1) | (df['NewExist'] == 2)]

df.loc[(df['NewExist'] == 1), 'NewBusiness'] = 0
df.loc[(df['NewExist'] == 2), 'NewBusiness'] = 1


# In[39]:


df.NewExist.unique()


# In[40]:


df.RevLineCr.unique()


# In[41]:


df.LowDoc.unique()


# In[42]:


df = df[(df.RevLineCr == 'Y') | (df.RevLineCr == 'N')]
df = df[(df.LowDoc == 'Y') | (df.LowDoc == 'N')]

df['RevLineCr'] = np.where(df['RevLineCr'] == 'N', 0, 1)
df['LowDoc'] = np.where(df['LowDoc'] == 'N', 0, 1)


# In[43]:


df.RevLineCr.unique()
df.LowDoc.unique()


# In[44]:


#Loan Status: Paid in Full, Charged Off


# In[45]:


df.MIS_Status.unique()


# In[46]:


df.MIS_Status.value_counts()


# In[47]:


df['Default'] = np.where(df['MIS_Status'] == 'P I F', 0, 1)
df['Default'].value_counts()


# In[48]:


df[['ApprovalDate', 'DisbursementDate']] = df[['ApprovalDate', 'DisbursementDate']].apply(pd.to_datetime)


# In[49]:


df['DaysToDisbursement'] = df['DisbursementDate'] - df['ApprovalDate']


# In[50]:


df.DaysToDisbursement.info()


# In[51]:


df['DaysToDisbursement'] = df['DaysToDisbursement'].astype('str').apply(lambda x: x[:x.index('d') - 1]).astype('int64')


# In[ ]:





# In[52]:


df['DisbursementFY'] = df['DisbursementDate'].map(lambda x: x.year)


# In[53]:


df['StateSame'] = np.where(df['State'] == df['BankState'], 1, 0)


# In[54]:


df['SBA_AppvPct'] = df['SBA_Appv'] / df['GrAppv']


# In[55]:


df['AppvDisbursed'] = np.where(df['DisbursementGross'] == df['GrAppv'], 1, 0)


# In[56]:


df.dtypes


# In[57]:


df = df.astype({'IsFranchise': 'int64', 'NewBusiness': 'int64'})


# In[58]:


df.dtypes


# In[59]:


df.drop(columns=['LoanNr_ChkDgt', 'Name', 'City', 'Zip', 'Bank', 'NAICS', 'ApprovalDate', 'NewExist', 'FranchiseCode',
                      'ChgOffDate', 'DisbursementDate', 'BalanceGross', 'ChgOffPrinGr', 'SBA_Appv', 'MIS_Status'], inplace=True)


# In[60]:


df.isnull().sum()


# In[61]:


df.shape


# In[62]:


df.Term.unique().sum()


# In[63]:


df['RealEstate'] = np.where(df['Term'] >= 240, 1, 0)


# In[64]:


df['GreatRecession'] = np.where(((2007 <= df['DisbursementFY']) & (df['DisbursementFY'] <= 2009)) | 
                                     ((df['DisbursementFY'] < 2007) & (df['DisbursementFY'] + (df['Term']/12) >= 2007)), 1, 0)


# In[65]:


df.DisbursementFY.unique()


# In[66]:


df = df[df.DisbursementFY <= 2010]


# In[67]:


df.shape


# In[68]:


#Analyzing the Data based on their dtypes


# In[69]:


df.describe(include = ['int', 'float', 'object'])


# In[70]:


df['DisbursedGreaterAppv'] = np.where(df['DisbursementGross'] > df['GrAppv'], 1, 0)


# In[71]:


df.DisbursedGreaterAppv.unique()


# In[72]:


df = df[df['DaysToDisbursement'] >= 0]

df.shape


# In[73]:


df.describe(include = ['int', 'float', 'object'])


# # Step-4 Data Visualization
# #Data Visualization
# #Here, the plan is to find out the correlation among the columns,
# #as well as find out the trends of the column, and
# #how the dataframe behaves based on the time frame, particularly the fluctuation during the Great Recession

# In[76]:


plt.figure(figsize=(20,10))
sns.heatmap(df.corr(), vmin=-1, vmax=1, cmap='GnBu', annot=True)
plt.show()


# In[79]:


plt.figure(figsize=(15,6))
ax = sns.histplot(df, x= df.State.astype(str), hue=df.Default.astype(str), palette='flare')
plt.title(label='Percentage of Defaulted Loans per State')
plt.ylabel('Number of Loans Given to Small Business')
plt.xticks(rotation=90,fontsize=7)
plt.show()


# In[ ]:





# In[81]:


plt.figure(figsize=(15,6))
ax = sns.histplot(df, x= df.GreatRecession.astype(str) ,weights=1, hue= df.Default.astype(str), palette='flare')
plt.title(label='Percentage of Loans Defaulted During Recession')
plt.ylabel('Number of Loans Given to Small Business')
plt.xlabel('Loans During Recession (1 - Rececssion / 0 - no Recession )')
plt.xticks(rotation=90,fontsize=7)
plt.show()


# In[82]:


industry_group = df.groupby(['Industry'])

df_industrySum = industry_group.sum().sort_values('DisbursementGross', ascending = False)
df_industryAve = industry_group.mean().sort_values('DisbursementGross', ascending=False)

fig = plt.figure(figsize=(40,20))

ax1 = fig.add_subplot(1, 2, 1)
ax1.bar(df_industrySum.index, df_industrySum['DisbursementGross'] / 1000000000)
ax1.set_xticklabels(df_industrySum.index, rotation=30, horizontalalignment='right', fontsize=10)

ax1.set_title('Gross SBA Loan Disbursement by Industry from 1984-2010', fontsize=30)
ax1.set_xlabel('Industry', fontsize = 30)
ax1.set_ylabel('Gross Loan Disbursement (Billions)', fontsize = 30)
plt.show()


# In[83]:


industry_group = df.groupby(['Industry'])

df_industrySum = industry_group.sum().sort_values('DisbursementGross', ascending = False)
df_industryAve = industry_group.mean().sort_values('DisbursementGross', ascending=False)

fig = plt.figure(figsize=(40,20))
ax2 = fig.add_subplot(1, 2, 2)
ax2.bar(df_industryAve.index, df_industryAve['DisbursementGross'])
ax2.set_xticklabels(df_industryAve.index, rotation=30, horizontalalignment='right', fontsize=10)

ax2.set_title('Average SBA Loan Disbursement by Industry from 1984-2010', fontsize=30)
ax2.set_xlabel('Industry',  fontsize = 30)
ax2.set_ylabel('Average Loan Disbursement',  fontsize = 30)

plt.show()


# In[84]:


#We notice, that Retail Trade and Manufacturing Industries have taken more loans than any other in this time period.

#But Agriculture, Forestry, Fishing, Hunting, Mining and more have small number of loans taken,
#but the amount of loan taken in total is the most relative to the other industries.


# In[85]:


fig2, ax = plt.subplots(figsize = (30,15))

ax.bar(df_industryAve.index, df_industryAve['DaysToDisbursement'].sort_values(ascending=False))
ax.set_xticklabels(df_industryAve['DaysToDisbursement'].sort_values(ascending=False).index, rotation=35,
                   horizontalalignment='right', fontsize=10)

ax.set_title('Average Days to SBA Loan Disbursement by Industry from 1984-2010', fontsize=15)
ax.set_xlabel('Industry')
ax.set_ylabel('Average Days to Disbursement')

plt.show()


# In[86]:


#Here, we notice that the industries with the highest avg loan amount also had the highest number of 
#days to disbursement of funds.

#Agri, Forestry, Fishing, Hunting ..


# In[ ]:





# In[87]:


fig3 = plt.figure(figsize=(50, 30))

ax1a = plt.subplot(2,1,1)
ax2a = plt.subplot(2,1,2)

def stacked_setup(df, col, axes, stack_col='Default'):
    data = df.groupby([col, stack_col])[col].count().unstack(stack_col)
    data.fillna(0)

    axes.bar(data.index, data[1], label='Default')
    axes.bar(data.index, data[0], bottom=data[1], label='Paid in full')

# Number of Paid in full and defaulted loans by industry
stacked_setup(df=df, col='Industry', axes=ax1a)
ax1a.set_xticklabels(df.groupby(['Industry', 'Default'])['Industry'].count().unstack('Default').index,
                     rotation=35, horizontalalignment='right', fontsize=10)

ax1a.set_title('Number of PIF/Defaulted Loans by Industry from 1984-2010', fontsize=50)
ax1a.set_xlabel('Industry')
ax1a.set_ylabel('Number of PIF/Defaulted Loans')
ax1a.legend()

# Number of Paid in full and defaulted loans by State
stacked_setup(df=df, col='State', axes=ax2a)

ax2a.set_title('Number of PIF/Defaulted Loans by State from 1984-2010', fontsize= 50)
ax2a.set_xlabel('State')
ax2a.set_ylabel('Number of PIF/Defaulted Loans')
ax2a.legend()

plt.tight_layout()
plt.show()


# In[88]:


def_ind = df.groupby(['Industry', 'Default'])['Industry'].count().unstack('Default')
def_ind['Def_Percent'] = def_ind[1]/(def_ind[1] + def_ind[0])

def_ind


# In[89]:


def_state = df.groupby(['State', 'Default'])['State'].count().unstack('Default')
def_state['Def_Percent'] = def_state[1]/(def_state[1] + def_state[0])

def_state


# In[90]:


fig4, ax4 = plt.subplots(figsize = (30,15))

stack_data = df.groupby(['DisbursementFY', 'Default'])['DisbursementFY'].count().unstack('Default')

x = stack_data.index
y = [stack_data[1], stack_data[0]]

ax4.stackplot(x, y, labels = ['Default', 'Paid In Full'])
ax4.set_title('Number of PIF/Defaulted Loans by State from 1984-2010', fontsize = 30)

ax4.set_xlabel('Disbursement Year')
ax4.set_ylabel('Number of PIF/Defaulted Loans')
ax4.legend(loc='upper left', fontsize = 20)

plt.show()


# # Step-5 Model Training and Testing
# #Here, the plan is to one hot encode the dataframe, 
# #Normalise the dataframe by scaling it and spliting the dataset into training and testing dataframes,
# #and train the model on the training dataset and test it on the testing and comparing the prediction and 
# #the testing target column using various metrics to find out the best possible model for the dataset.
# 
# #The Classifier Models to be used are:
# 
# #Logistic Regression
# #Decision Tree Classifier
# #Random Forest Classifier
# #Naive Bayes
# #Voting Classifier
# #Linear Discriminant Analysis

# In[92]:


df = pd.get_dummies(df)

df.head()


# In[93]:


y = df['Default']
X = df.drop('Default', axis = 1)


# In[94]:


scale = StandardScaler()
X_scld = scale.fit_transform(X)


# In[95]:


X_train, X_val, y_train, y_val = train_test_split(X_scld, y, test_size=0.25,random_state = 7)


# # Logistic Regression

# In[97]:


from sklearn.metrics import classification_report


# In[98]:


lr = LogisticRegression()

lr.fit(X_train, y_train)
y_pred = lr.predict(X_val)

print(classification_report(y_val, y_pred, digits = 3))


# In[99]:


# Confusion Matrix
cm = confusion_matrix(y_val, y_pred, labels=lr.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lr.classes_)
disp.plot()


# # Decision Tree

# In[101]:


dtc = DecisionTreeClassifier()
model_dtc = dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_val)

print(classification_report(y_val, y_pred, digits = 3))


# In[102]:


# Confusion Matrix
cm = confusion_matrix(y_val, y_pred, labels=dtc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dtc.classes_)
disp.plot()


# In[103]:


print("Testing accuracy is:",accuracy_score(y_val, y_pred))


# # Random Forest Classifier

# In[133]:


rfc = RandomForestClassifier()
model_rfc = rfc.fit(X_train, y_train)

predictions= rfc.predict(X_val)

print(classification_report(y_val, predictions, digits = 3))


# In[134]:


# Confusion Matrix
cm = confusion_matrix(y_val, predictions, labels=rfc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rfc.classes_)
disp.plot()


# In[135]:


print("Testing accuracy is:",accuracy_score(y_val, predictions))


# # Naive Bayes 

# In[110]:


gnb = GaussianNB()
model_gnb = gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_val)

print(classification_report(y_val, y_pred, digits = 3))


# In[111]:


# Confusion Matrix
cm = confusion_matrix(y_val, y_pred, labels=gnb.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gnb.classes_)
disp.plot()


# In[112]:


print("Testing Accuracy is: ", accuracy_score(y_val, y_pred))


# # Support Vector Classifier

# In[ ]:


model=SVC()
model_svc=model.fit(X_train, y_train)

y_pred=model.predict(X_val)

print(classification_report(y_val, y_pred, digits = 3))


# In[ ]:


print("Testing Accuracy is: ", accuracy_score(y_val, y_pred))


# In[143]:


# It is taking huge amount of time to run.


# # LinearDiscriminantAnalysis

# In[115]:


model=LinearDiscriminantAnalysis()
model_LDA=model.fit(X_train, y_train)

y_pred=model.predict(X_val)

print(classification_report(y_val, y_pred, digits = 3))


# In[116]:


# Confusion Matrix
cm = confusion_matrix(y_val, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()


# In[117]:


print("Testing Accuracy is: ", accuracy_score(y_val, y_pred))


# # Voting Classifier

# In[119]:


from sklearn.ensemble import VotingClassifier
rfc = RandomForestClassifier(random_state=42)
dtc = DecisionTreeClassifier(random_state=42)
lr = LogisticRegression()

pipe = VotingClassifier([('dtc', dtc),('rfc', rfc),('lr', lr)], weights = [4,5,1])


# In[120]:


pipe.fit(X_train, y_train)


# In[121]:


y_pred = pipe.predict(X_val)
print(classification_report(y_val, y_pred, digits = 3))


# In[122]:


# Confusion Matrix
cm = confusion_matrix(y_val, y_pred, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()


# In[123]:


print("Testing Accuracy is: ", accuracy_score(y_val, y_pred))


# # KNearestneighbors

# In[ ]:


knn=KNeighborsClassifier()
model_knn=knn.fit(X_train, y_train)

y_pred=knn.predict(X_val)

print(classification_report(y_val, y_pred, digits = 3))


# In[ ]:


print("Testing Accuracy is: ", accuracy_score(y_val, y_pred))


# In[ ]:


# It is taking huge amount of time to run. But the code is correct.


# # XGBoost model

# In[146]:


xgboost = XGBClassifier(random_state=2)

xgboost.fit(X_train, y_train)
y_xgbpred = xgboost.predict(X_val)

# Print the results
print(classification_report(y_val, y_xgbpred, digits=3))


# In[147]:


# List the importance of each feature
for name, importance in sorted(zip(X.columns, xgboost.feature_importances_)):
    print(name, "=", importance)


# In[148]:


# Build pipeling for feature selection and modeling; SelectKBest defaults to top 10 features
xgb_featimp = XGBClassifier(random_state=2)

pipe = Pipeline(steps=[
    ('feature_selection', SelectKBest()),
    ('model', xgb_featimp)
])

pipe.fit(X_train, y_train)
y_featimppred = pipe.predict(X_val)

print(classification_report(y_val, y_featimppred, digits=3))


# In[149]:


# List the importance of each feature
for name, importance in sorted(zip(X.columns, xgb_featimp.feature_importances_)):
    print(name, "=", importance)


# # Step-5:- Save and Load the Machine Learning Models : (2 Methods)
# #--------------------------------------------------------
# #This allows you to save your model to file and load it later
# #in order to make predictions in future.
# 
# #Method-1) Use pickle to serialize(=dump) and deserialize(=load)
# # #       machine learning models.
# 
# #Method-2) Use Joblib to serialize(=dump) and deserialize(=load)
# # #       machine learning models.

# # I am using Method-1)

# In[150]:


import pickle


# In[151]:


#save the model to disk
filename = "C://Users//Narendra Malviya//Desktop//Unguided Project//Finalized_Model.sav"
pickle.dump(model,open(filename, "wb"))     #wb= write binary

print("Model dumped successfully into a file by Pickle"
     ".....\n....\n....\n....")

print("-------------------------\n\n\n")
print("some time later....   ")
print("\n\n\n-------------------")


# In[152]:


#load the model from disk
import pickle
filename = "C://Users//Narendra Malviya//Desktop//Unguided Project//Finalized_Model.sav"
loaded_model = pickle.load(open(filename, "rb"))     #rb = reuse binary
print("Model loaded successfully from the file by Pickle")


# # In conclusion, we get the best result from the Random Forest Classifier and XGBoost Classifier with an accuracy score of 94.6 and 94.5 on the testing dataframe respectively.
# 
# #With this project, I got to learn about the Loan Approval Process for Small Business Administration (SBA), 
# #as well about the entire process from taking the loan to disbursement of it, based on the different sectors of business. 
# #Also understood the effect of Great Recession 
# #on the Disbursement of the loans during that period, and it's effect on taking loans.

# # Submitted By:- Nimisha Malviya
# The Unguided Project Of Artificial Intelligence Batch-1(18May-15June) 

# # The Unguided Project Of Artificial Intelligence Batch-1(18May-15June)

# In[ ]:




