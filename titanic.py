import pandas
from sklearn.linear_model import LinearRegression


titanic=pandas.read_csv("~/Downloads/train.csv")
#print(titanic.head(5))
titanic['Age']=titanic['Age'].fillna(titanic['Age'].median())
titanic.loc[titanic['Sex']=='male','Sex']=0
titanic.loc[titanic['Sex']=='female','Sex']=1

titanic['Embarked']=titanic['Embarked'].fillna('S')
titanic.loc[titanic['Embarked']== 'S','Embarked']=0
titanic.loc[titanic['Embarked']=='C','Embarked']=1
titanic.loc[titanic['Embarked']=='Q','Embarked']=2
#print(titanic['Embarked'].unique())
import numpy as np
#from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
alg= LogisticRegression(random_state=1)
#kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

"""for train, test in kf:
    train_predictors = (titanic[predictors].iloc[train, :])
    train_target = titanic["Survived"].iloc[train]
    algorithm.fit(train_predictors, train_target)
    test_predictions = algorithm.predict(titanic[predictors].iloc[test, :])
    predictions.append(test_predictions)"""




titanic_test = pandas.read_csv("~/Downloads/test.csv")
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

alg.fit(titanic[predictors],titanic['Survived'])

predictions = alg.predict(titanic_test[predictors])

submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })

from sklearn.metrics import accuracy_score

print(accuracy_score(predictions,titanic_test['Survived']))

# submission.to_csv("~/Desktop/titanic.csv", index=False)
#print(titanic.head(5))
#print(titanic.describe())