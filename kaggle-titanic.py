import warnings
warnings.filterwarnings(action='ignore')  #no print warning
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

#get title from 
def getTitle(data):
    for i in data.index:
        splits= data['Name'][i].split(",")
        splits2=splits[1].split(".")
        data['Name'][i]=splits2[0].strip()
    
    for i in data.index:
        if  data['Name'][i]=='Mlle':
            data['Name'][i] = data['Name'][i].replace('Mlle', 'Miss')
        elif data['Name'][i] =='Mme':
            data['Name'][i] = data['Name'][i].replace('Mme', 'Mrs')
        elif data['Name'][i] =='Ms':
            data['Name'][i] = data['Name'][i].replace('Ms', 'Miss')
        elif data['Name'][i] not in ['Mr', 'Mrs', 'Miss']:
            data['Name'][i] = 'Other'

#categorize age
def changeAgeData(data) :
    for i in data.index:
        if data['Age'][i] <= 12 :
            data['Age'][i]=0

        elif data['Age'][i] <=20 :
            data['Age'][i]=1

        elif data['Age'][i] <= 40 :
            data['Age'][i]=2

        else:
            data['Age'][i]=3

#calculate # of family
def getFamily(data):
    data['Family']=0;
    for i in data.index:
        data['Family'][i] = data['Parch'][i] + data['SibSp'][i]
        data['Family'][i] = data['Family'][i].astype(int)
        
#categorize fare   
def changeFareData(data):
    data['Fare_bin']=0
    data['Fare_bin']= pd.cut(data['Fare'], bins=[0,7.91,14.45,31,120], labels=['Low_fare','median_fare',
                                                                                      'Average_fare','high_fare'])
#data load
data_titanic=pd.read_csv('./train.csv', delimiter=',')
data_titanic_test=pd.read_csv('./test.csv', delimiter=',')

#change sex to num
change_value_dict={'female':0, 'male':1}
data_titanic=data_titanic.replace({'Sex':change_value_dict})
data_titanic_test=data_titanic_test.replace({'Sex':change_value_dict})

#embarked missing value
data_titanic['Embarked'] = data_titanic['Embarked'].fillna(data_titanic['Embarked'].mode())

#change embarked to num
data_titanic=pd.get_dummies(data_titanic, columns=['Embarked'])
data_titanic_test=pd.get_dummies(data_titanic_test, columns=['Embarked'])

#get title from name
getTitle(data_titanic)
getTitle(data_titanic_test)
data_titanic.rename(columns = {'Name' : 'Title'}, inplace = True)
data_titanic_test.rename(columns = {'Name' : 'Title'}, inplace = True)

#age missing value
data_titanic['Age'].fillna(data_titanic.groupby('Title')['Age'].transform('median'), inplace=True)
data_titanic_test['Age'].fillna(data_titanic_test.groupby('Title')['Age'].transform('median'), inplace=True)

#categorize age
changeAgeData(data_titanic)
changeAgeData(data_titanic_test)

#get # of family
getFamily(data_titanic)
getFamily(data_titanic_test)

#categorize fare
changeFareData(data_titanic)

#one-hot encoding:title, far_bin
data_titanic=pd.get_dummies(data_titanic, columns=['Title'])
data_titanic=pd.get_dummies(data_titanic, columns=['Fare_bin'])
data_titanic_test=pd.get_dummies(data_titanic_test, columns=['Title'])

#drop unused col
data_titanic=data_titanic.drop(['Cabin','Ticket', 'SibSp', 'Parch', 'Fare'], axis=1)

#train data
X_train=data_titanic.loc[:,'Pclass':'Fare_bin_high_fare']
Y_train=data_titanic['Survived']


#random forest
rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(X_train, Y_train)

#extract passengerId to use making submission.csv
passenger=data_titanic_test['PassengerId']

#fare missing value, categorize, one-hot enconding in test data
fare=data_titanic_test['Fare'][data_titanic['Pclass']==3].mean()
data_titanic_test['Fare']=data_titanic_test['Fare'].fillna(fare)
changeFareData(data_titanic_test)
data_titanic_test=pd.get_dummies(data_titanic_test, columns=['Fare_bin'])

#drop unuse col
data_titanic_test=data_titanic_test.drop(['Cabin','Ticket', 'SibSp', 'Parch', 'Fare'], axis=1)

#random forest
pred=rf_clf.predict(data_titanic_test.loc[:,'Pclass':'Fare_bin_high_fare'])

#save output as submission.csv
series_pred = pd.Series(pred)
submission=pd.concat([passenger, series_pred], axis=1)
submission.rename(columns = {0 : 'Survived'}, inplace = True)
submission['PassengerId'] = submission['PassengerId'].astype(int)
submission.to_csv('submission.csv', index=False)
 
