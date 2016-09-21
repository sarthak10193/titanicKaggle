import pandas as pd
from pandas import Series, DataFrame

import seaborn as sns
sns.set(style="darkgrid", color_codes = True)

import numpy as np

import matplotlib.pyplot as plt

import scipy as sp
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score

# Reading the data
entire_titanic_train = pd.read_csv("/home/sarthak/PycharmProjects/Titanic_Kaggle/data/train.csv")
titanic_test = pd.read_csv("/home/sarthak/PycharmProjects/Titanic_Kaggle/data/test.csv")

# getting the cross validation set
msk = np.random.rand(len(entire_titanic_train)) < 0.75
titanic_train = entire_titanic_train[msk]
titanic_cv = entire_titanic_train[~msk]

print("Shape of training data : ",titanic_train.shape , "\n")
# print(" --- Data head ---  \n" ,titanic_train.head() , "\n")
# print(" ---  data info --- \n" , titanic_train.info(), "\n")
print("Survived count " , titanic_train["Survived"][titanic_train["Survived"]==1].count())
print("NOT survived count " , titanic_train["Survived"][titanic_train["Survived"]==0].count())
print(titanic_train.iloc[:2,[2,3,6]])

# drop uneccessary columns which won't be useful to predict the survival
titanic_train = titanic_train.drop(['Name', 'Ticket'], axis=1)
titanic_test = titanic_test.drop(['Name', 'Ticket'], axis=1)
titanic_cv = titanic_cv.drop(['Name', 'Ticket'], axis=1)

######################### P class #####################
print("\n\n----------------------- Pclass --------------------------\n\n")
print(titanic_train["Pclass"].unique())
print(titanic_train["Pclass"].isnull().sum())

print(pd.value_counts(titanic_train["Pclass"].values , sort=False))
pclass_survived = titanic_train["Pclass"][titanic_train["Survived"]==1]
g1 = sns.factorplot(x='Pclass',y='Survived',order=[1,2,3], data=titanic_train,size=5 ,kind="point", legend=True , hue="Pclass")
g1.set_axis_labels(x_var="Class Varialbes [1,2,3]" ,y_var="% Survived")
sns.plt.title("plotting three classes against survived ")

# we observe that survival varies quite a bit depending on the class of the passenger
pclass_titanic_train_dummy = pd.get_dummies(titanic_train["Pclass"])
pclass_titanic_train_dummy.columns = ['Class_1', 'Class_2', 'Class_3']
titanic_train = titanic_train.join(pclass_titanic_train_dummy)

print("hello")
pclass_titanic_test_dummy = pd.get_dummies(titanic_test['Pclass'])
pclass_titanic_test_dummy.columns = ['Class_1', 'Class_2', 'Class_3']
titanic_test = titanic_test.join(pclass_titanic_test_dummy)

pclass_titanic_cv_dummy = pd.get_dummies(titanic_cv['Pclass'])
pclass_titanic_cv_dummy.columns = ['Class_1', 'Class_2', 'Class_3']
titanic_cv = titanic_cv.join(pclass_titanic_cv_dummy)

titanic_train.drop(['Pclass'], axis=1, inplace=True)
titanic_test.drop(['Pclass'], axis=1, inplace=True)
titanic_cv.drop(['Pclass'], axis=1, inplace=True)

############################ EMBARKED #########################
print("----------------------- EMBARKED --------------------------\n\n")
print("No. of null values in Embarked_train : " , titanic_train["Embarked"].isnull().sum())
print(titanic_train["Embarked"].unique())
print(pd.value_counts(titanic_train["Embarked"].values , sort = False))

# replacing NAN in e mbarked with the most frequent clas
titanic_train["Embarked"] = titanic_train["Embarked"].fillna('S', inplace=False)

sns.factorplot(x='Embarked',y='Survived', data=titanic_train,size=4,aspect=3)
sns.plt.title("factor plot for embarked vs the % survived for each")

fig, (axis1, axis2, axis3) = plt.subplots(1,3,figsize=(3,15))
g1 = sns.countplot(x='Embarked', data=titanic_train , ax=axis1)
g2 = sns.countplot(x='Survived', hue='Embarked', ax=axis2, data=titanic_train)
'''
here we take the ratio of embarked values {S,Q,C} to Survival=1  ... same thing as factorplot above
just a different form of display
'''
g3 = sns.factorplot(x='Embarked', y='Survived', kind='point', ax=axis3, data=titanic_train, hue="Embarked")
fig.suptitle("Fig 1. plotting all embarked count values {S,Q,C} \n"
             "Fig 2. plotting embarked values for Suv = 0/1 \n"
             "Fig 3 Plotting the ratio of Embarked values to the respective total survived")


"""
seeing the plot I infer that C and Q could be important parameters in determining the survival,
and S has little role to play
Create dummy values for {S,Q,C} and then delete S dummy values and join the rest to the original DF
"""
titanic_train_dummy_emb = pd.get_dummies(titanic_train["Embarked"])
titanic_train = titanic_train.join(titanic_train_dummy_emb)
titanic_train.drop(['Embarked'], axis=1, inplace=True)

titanic_test_dummy_emb = pd.get_dummies(titanic_test['Embarked'])
titanic_test = titanic_test.join(titanic_test_dummy_emb)
titanic_test.drop(['Embarked'],axis=1, inplace=True)

titanic_cv_dummy_emb = pd.get_dummies(titanic_cv['Embarked'])
titanic_cv = titanic_cv.join(titanic_cv_dummy_emb)
titanic_cv.drop(['Embarked'],axis=1, inplace=True)

########################## Fare Dependecy ##############
print("----------------------- Fare --------------------------\n\n")
print("no of null values in Fares_train : " , titanic_train["Fare"].isnull().sum())
print("no of null values in Fares_test : " , titanic_test["Fare"].isnull().sum())
print("no of null values in Fares_cv : " , titanic_cv["Fare"].isnull().sum())

titanic_test["Fare"].fillna(titanic_test["Fare"].median(), inplace=True)
# split into survived and not survived with respect to fares
titanic_train_fares_survived = titanic_train["Fare"][titanic_train["Survived"]==1]
titanic_train_fares_notsurvived = titanic_train["Fare"][titanic_train["Survived"]==0]

average_fare = DataFrame([titanic_train_fares_notsurvived.mean(), titanic_train_fares_survived.mean()])
std_fare = DataFrame([titanic_train_fares_notsurvived.std(), titanic_train_fares_survived.std()])
print(average_fare.shape)

average_fare.index.names = std_fare.index.names = ['Survived']
g = sns.boxplot(x="Survived" , y = 'Fare' , hue='Survived' , data=titanic_train ,linewidth= 1.0)
plt.xlabel("Fare amount")
plt.ylabel("Survived value")
plt.title("boxplot showing average, min, max for fare vs survived")

# plotting the box plot we see a considerable difference btw average fares for survived 0 and 1

# ######################### Cabin ########################
print("----------------------- Cabin --------------------------\n\n")
print("no of null values in Cabin : " , titanic_train["Cabin"].isnull().sum())
"""
Too many null values , therefore isn't of any importance to predict survival , drop the entire column
"""
titanic_train.drop(["Cabin"], axis=1, inplace=True)
titanic_test.drop(["Cabin"], axis=1, inplace=True)
titanic_cv.drop(["Cabin"], axis=1, inplace=True)

# ######################### Age ########################
print("----------------------- Age --------------------------\n\n")

titanic_train_avgAge = titanic_train["Age"].mean()
titanic_train_stdAge = titanic_train["Age"].std()     # standard deviation
titanic_train_nullAgeCount = titanic_train["Age"].isnull().sum()

titanic_test_avgAge = titanic_test['Age'].mean()
titanic_test_stdAge = titanic_test['Age'].std()
titanic_test_nullAgeCount = titanic_test['Age'].isnull().sum()

"""
#for the missing 167 values of Age , one good way to fill them up is generate random values
#between mean-1std and mean+1std
"""
titanic_train_randAge = np.random.randint(titanic_train_avgAge-titanic_train_stdAge,
                                            titanic_train_avgAge+titanic_train_stdAge,
                                            size= titanic_train_nullAgeCount)
titanic_test_randAge = np.random.randint(titanic_test_avgAge-titanic_test_stdAge,
                                         titanic_test_avgAge+titanic_test_stdAge,
                                         size=titanic_test_nullAgeCount)

# drop na values and plot the original values in a histogram
fig, (axis1, axis2)= plt.subplots(1, 2, sharex=True)

axis1.hist(titanic_train["Age"].dropna().astype(int), bins=100,color="blue", stacked=False)

titanic_train["Age"][np.isnan(titanic_train["Age"])] = titanic_train_randAge
titanic_test['Age'][np.isnan(titanic_test['Age'])] = titanic_test_randAge

axis2.hist(titanic_train["Age"].astype(int), bins=100, color="green")

plt.xlabel("Age in intervals")
plt.ylabel("Frequency")
fig.suptitle("1. original age distribution (removing nan) 2. age distribution with newly added values")
# so we observe we did a good job generating the new values for age since the two curves have similar distribution

facet = sns.FacetGrid(titanic_train, hue="Survived",aspect=4)
facet.map(sns.kdeplot, 'Age', shade= True)
facet.set(xlim=(0, titanic_train['Age'].max()))
facet.add_legend()

####################### Sex #####################
print("----------------------- Sex --------------------------\n\n")
print("unique sex values : " , titanic_train["Sex"].unique())
print("no of null values for sex:" , titanic_train["Sex"].isnull().sum())

# we saw that people under age<16 are very likely to survive , so lets create three types of peopel
# men , women and children
def get_person(passenger):
    age, sex = passenger
    if age < 16:
        return 'child'
    else:
        return sex

titanic_train['Person'] = titanic_train[['Age', 'Sex']].apply(get_person, axis = 1)
titanic_train_dummy_person = pd.get_dummies(titanic_train['Person'])
titanic_train = titanic_train.join(titanic_train_dummy_person)

titanic_test['Person'] = titanic_test[['Age','Sex']].apply(get_person, axis=1)
titanic_test_dummy_person = pd.get_dummies(titanic_test['Person'])
titanic_test = titanic_test.join(titanic_test_dummy_person)

titanic_cv['Person'] = titanic_cv[['Age','Sex']].apply(get_person, axis=1)
titanic_cv_dummy_person = pd.get_dummies(titanic_cv['Person'])
titanic_cv = titanic_cv.join(titanic_cv_dummy_person)


# drop the sex and person column
titanic_train.drop(['Person'], axis = 1, inplace=True)
titanic_train.drop(['Sex'], inplace=True, axis=1)

titanic_test.drop(['Person'], axis = 1, inplace=True)
titanic_test.drop(['Sex'], inplace=True, axis=1)

titanic_cv.drop(['Person'], axis = 1, inplace=True)
titanic_cv.drop(['Sex'], inplace=True, axis=1)


###################### Family #####################

print("----------------------- Family --------------------------\n\n")
print(titanic_train["Parch"].unique())
print(titanic_train["SibSp"].unique())

titanic_train['Family'] = titanic_train["Parch"] + titanic_train["SibSp"]
titanic_train['Family'].loc[titanic_train['Family'] > 0] = 1   # has family
titanic_train['Family'].loc[titanic_train['Family'] == 0] = 0  # no family

# family_member_values = titanic_train['Family'].unique()
titanic_train.drop(['Parch'], inplace=True, axis=1 )
titanic_train.drop(['SibSp'], inplace=True, axis=1 )


fig , (axis1, axis2) = plt.subplots(1,2,sharex=True , figsize=(10,5))
# sns.countplot(x='Family', data=titanic_train, order=sorted(family_member_values), ax=axis1)
sns.countplot(x='Family', data=titanic_train, order=[0,1], ax=axis1)
sns.factorplot(x='Family', y='Survived', kind='point', ax=axis2, size=5, legend=True, data=titanic_train, hue='Family')

# operating the same on test data
titanic_test['Family'] = titanic_test['Parch'] + titanic_test['SibSp']
titanic_test['Family'].loc[titanic_test['Family'] == 0] = 0  # no family
titanic_test['Family'].loc[titanic_test['Family'] > 0] = 1   # has family

titanic_test.drop(['Parch'], inplace=True, axis=1)
titanic_test.drop(['SibSp'], inplace=True, axis=1)

# operating the same on cross-validation set
titanic_cv['Family'] = titanic_cv['Parch'] + titanic_cv['SibSp']
titanic_cv['Family'].loc[titanic_cv['Family'] == 0] = 0  # no family
titanic_cv['Family'].loc[titanic_cv['Family'] > 0] = 1   # has family

titanic_cv.drop(['Parch'], inplace=True, axis=1)
titanic_cv.drop(['SibSp'], inplace=True, axis=1)

"""
# splitting family col into alone, nuclear and large-family
titanic_train['Family0'] = titanic_train['Family'].loc[titanic_train['Family'] == 0]
titanic_train['Family0'][titanic_train['Family0'] == 0] = 1
titanic_train['Family0'].fillna(0, inplace=True)

titanic_train['Family123'] = titanic_train['Family'].loc[(titanic_train['Family'] == 1)]

i = 0
for x in titanic_train['Family']:
    if (x == 1) or (x==2) or (x==3):
        i += 1
        titanic_train['Family123'][i] = 1


titanic_train['Family123'].fillna(0, inplace=True)

titanic_train['Family>3'] = titanic_train['Family'].loc[titanic_train['Family'] > 3]
titanic_train['Family>3'][titanic_train['Family>3'] > 3] =1
titanic_train['Family>3'].fillna(0, inplace=True)

titanic_train.drop(['Family'], axis=1)  # now you can drop family too

print("Family 0 : ", pd.value_counts(titanic_train['Family0'].values),"\n" ,
       "Family123 :  ", pd.value_counts(titanic_train['Family123'].values), "\n",
      "Family>3 : ",pd.value_counts(titanic_train['Family>3'].values))

"""

print('################### END ######################')
'''
# the feature important for the classification problem
PassengerId    685 non-null int64
Survived       685 non-null int64

1. Age            685 non-null float64
2. Fare           685 non-null float64
3. Class_1        685 non-null float64
4. Class_2        685 non-null float64
5. Class_3        685 non-null float64
6. C              685 non-null float64
7. Q              685 non-null float64
8. S              685 non-null float64
9. child          685 non-null float64
10. female         685 non-null float64
11. male           685 non-null float64
12. Family         685 non-null int64

[[-0.01658916  0.00186222  1.14151935  0.29646607 -0.9664262   0.46273319
   0.28853812 -0.27971208  0.63377929  1.30037494 -1.462595   -0.10072113]]

PassengerId    418 non-null int64

Age            332 non-null float64
Fare           417 non-null float64
Class_1        418 non-null float64
Class_2        418 non-null float64
Class_3        418 non-null float64
C              418 non-null float64
Q              418 non-null float64
S              418 non-null float64
child          418 non-null float64
female         418 non-null float64
male           418 non-null float64
Family         418 non-null int64


'''
# training the model
X = titanic_train.drop(['Survived','PassengerId'],axis=1)
Y = titanic_train['Survived']
log_regression = LogisticRegression()
log_regression.fit(X,Y)

# testing the test set for the given model
X_test = titanic_test.drop(['PassengerId'], axis=1)
Y_pred = log_regression.predict(X_test)
print("shape of o/p vector : ",  Y_pred.shape , "Shape of test data : ", titanic_test.shape)
print("\n ----- predicted values  -----  \n", Y_pred)

Y_test = pd.read_csv("/home/sarthak/PycharmProjects/Titanic_Kaggle/data/gendermodel.csv")

# note : training set error is not a good metric for evaluating the model , too low error could be overfitting
print("training set score : ", log_regression.score(X,Y))

accuracy = accuracy_score(Y_test['Survived'], Y_pred)
print("error  = ", 1-accuracy, "accuracy  = ", accuracy)

# submission
np.savetxt("/home/sarthak/PycharmProjects/Titanic_Kaggle/data/mypred.csv", Y_pred, delimiter=",")



plt.plot(x=[12,3,4],  y = [1,2,3])
plt.show()









