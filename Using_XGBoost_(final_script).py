import pandas as pd
from sklearn.cross_validation import train_test_split
from xgboost.sklearn import XGBClassifier
import math
import numpy as np

# load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# pre-processing
def clean_train(train):
    # map animal types with integers (0 and 1)
    animal_type = {'Cat': 0, 'Dog': 1}
    train['AnimalType'] = train['AnimalType'].map(animal_type)

    # time of the day
    train['Hour'] = np.exp(train['DateTime'].str[11:13].astype(float))
    train['Minute'] = train['DateTime'].str[14:16].astype(int)

    # create weekday attribute
    train['DateTime'] = pd.to_datetime(train['DateTime'])
    train['WeekDay'] = train['DateTime'].dt.dayofweek

    # create month attribute
    train['Month'] = train['DateTime'].dt.month

    # create year attribute
    train['Year'] = train['DateTime'].dt.year

    # outcome
    outcome = {'Adoption': 0, 'Died': 1, 'Euthanasia': 2, 'Return_to_owner': 3, 'Transfer': 4}
    train['OutcomeType'] = train['OutcomeType'].map(outcome)

    # age
    def years(age):
        try:
            x = age.split()
        except:
            return None
        if (x[1] == 'years' or x[1] == 'year'):
            return int(x[0])
        else:
            return None

    train['AgeYears'] = train['AgeuponOutcome'].map(years)

    def months(age):
        try:
            x = age.split()
        except:
            return None
        if (x[1] == 'months' or x[1] == 'month'):
            return x[0]
        else:
            return None

    train['AgeMonths'] = train['AgeuponOutcome'].map(months)

    def weeks(age):
        try:
            x = age.split()
        except:
            return None
        if (x[1] == 'weeks' or x[1] == 'week'):
            return x[0]
        else:
            return None

    train['AgeWeeks'] = train['AgeuponOutcome'].map(weeks)

    def days(age):
        try:
            x = age.split()
        except:
            return None
        if (x[1] == 'days' or x[1] == 'day'):
            return x[0]
        else:
            return None

    train['AgeDays'] = train['AgeuponOutcome'].map(days)

    def Age(age):
        try:
            x = age.split()
        except:
            return 0
        if 'day' in x[1]:
            return math.tanh(int(x[0]))
        elif 'week' in x[1]:
            return math.tanh(int(x[0]) * 7)
        elif 'month' in x[1]:
            return math.tanh(int(x[0]) * 30)
        elif 'year' in x[1]:
            return math.tanh(int(x[0]) * 365 + 1)

    train['Age'] = train['AgeuponOutcome'].map(Age)
    train.loc[(train['Age'] == 0), 'Age'] = train['Age'].mean()
    train['Age'] = train['Age'].astype(int)

    # sex
    def Type(string):
        try:
            x = string.split()
        except:
            return None
        if len(x) == 1:
            return None
        if x[0] == 'Intact':
            return True
        else:
            return False

    train['Type'] = train['SexuponOutcome'].map(Type)

    def Gender(string):
        try:
            x = string.split()
        except:
            return None
        if len(x) == 1:
            return None
        elif x[1] == 'Male':
            return True
        else:
            return False

    train['Gender'] = train['SexuponOutcome'].map(Gender)


    # name
    def isName(name):
        try:
            x = len(name)
        except:
            return False
        return True

    train['isName'] = train['Name'].map(isName)

    # shades
    def isShade(color):
        shades = ["Merle", "Brindle", "Tiger", "Smoke", "Cream", "Point", "Tick", "Tabby"]
        try:
            x = color.split('/')
        except:
            y = color.split()
            if len(y) == 1:
                return False
            else:
                if y[-1] in shades:
                    return True
                else:
                    return False
        for item in x:
            y = item.split()
            if y[-1] in shades:
                return True
            else:
                return False

    train['isShade'] = train['Color'].map(isShade)

    shades = ["Merle", "Brindle", "Tiger", "Smoke", "Cream", "Point", "Tick", "Tabby"]
    def Shade(color):
        try:
            x = color.split('/')
        except:
            y = color.split()
            if len(y) == 1:
                return False
            else:
                if shade == y[-1]:
                    return True
                else:
                    return False
        for item in x:
            y = item.split()
            if shade == y[-1]:
                return True
            else:
                return False

    for shade in shades:
        train[shade] = train['Color'].map(Shade)

    def isShade(color):
        try:
            x = color.split('/')
        except:
            y = color.split()
            if len(y) == 1:
                return False
            else:
                if y[-1] in shades:
                    return True
                else:
                    return False
        for item in x:
            y = item.split()
            if y[-1] in shades:
                return True
            else:
                return False

    for shade in shades:
        train["isShade"] = train['Color'].map(isShade)

    # colors
    colors = []
    a = train.Color.str.split('/')
    b = test.Color.str.split('/')
    ab = a.append(b)
    for i in ab:
        for c in i:
            colors.append(c)
    colors = set(colors)

    def isColor(color_s):
        try:
            x = color_s.split('/')
        except:
            if color == color_s:
                return True
            else:
                return False
        if color in x:
            return True
        else:
            return False

    for color in colors:
        train[color] = train['Color'].map(isColor)

    def Count(colors):
        try:
            x = colors.split('/')
        except:
            return 1
        if x:
            return len(x)

    train['NbColor'] = train['Color'].map(Count)

    # breeds
    breeds = []
    a = train.Breed.str.split('/')
    b = test.Breed.str.split('/')
    ab = a.append(b)
    for i in ab:
        for c in i:
            if c[-3:] == 'Mix':
                c = c[:-4]
                breeds.append(c)
            else:
                breeds.append(c)
    breeds = set(breeds)

    def isBreed(breed_s):
        if breed_s[-3:] == 'Mix':
            breed_s = breed_s[:-4]
            try:
                x = breed_s.split('/')
            except:
                if breed == breed_s:
                    return True
                else:
                    return False
            if breed in x:
                return True
            else:
                return False
        else:
            try:
                x = breed_s.split('/')
            except:
                if breed == breed_s:
                    return True
                else:
                    return False
            if breed in x:
                return True
            else:
                return False

    for breed in breeds:
        train[breed] = train['Breed'].map(isBreed)

    def Count_breed(breeds):
        try:
            x = breeds.split('/')
        except:
            return 1
        if x:
            return len(x)

    train['NbBreed'] = train['Breed'].map(Count_breed)

    def isMix(breeds):
        if breeds[-3:] == 'Mix':
            return True
        else:
            return False

    train['isMix'] = train['Breed'].map(isMix)

    # color + breed + mix
    train['ColorBreed'] = train['NbColor'] + train['NbBreed'] + train['isMix'] - 2

    # drop irrelevant attributes
    train.drop(['AnimalID', 'OutcomeSubtype', 'DateTime', 'Name'], axis=1, inplace=True)

    return train


###########
###########

def clean_test(test):
    # map animal types with integers (0 and 1)
    animal_type = {'Cat': 0, 'Dog': 1}
    test['AnimalType'] = test['AnimalType'].map(animal_type)

    # time of the day
    test['Hour'] = np.exp(test['DateTime'].str[11:13].astype(float))
    test['Minute'] = test['DateTime'].str[14:16].astype(int)

    # create weekday attribute
    test['DateTime'] = pd.to_datetime(test['DateTime'])
    test['WeekDay'] = test['DateTime'].dt.dayofweek

    # create month attribute
    test['Month'] = test['DateTime'].dt.month

    # create year attribute
    test['Year'] = test['DateTime'].dt.year

    # age
    def years(age):
        try:
            x = age.split()
        except:
            return None
        if (x[1] == 'years' or x[1] == 'year'):
            return int(x[0])
        else:
            return None

    test['AgeYears'] = test['AgeuponOutcome'].map(years)

    def months(age):
        try:
            x = age.split()
        except:
            return None
        if (x[1] == 'months' or x[1] == 'month'):
            return x[0]
        else:
            return None

    test['AgeMonths'] = test['AgeuponOutcome'].map(months)

    def weeks(age):
        try:
            x = age.split()
        except:
            return None
        if (x[1] == 'weeks' or x[1] == 'week'):
            return x[0]
        else:
            return None

    test['AgeWeeks'] = test['AgeuponOutcome'].map(weeks)

    def days(age):
        try:
            x = age.split()
        except:
            return None
        if (x[1] == 'days' or x[1] == 'day'):
            return x[0]
        else:
            return None

    test['AgeDays'] = test['AgeuponOutcome'].map(days)

    def Age(age):
        try:
            x = age.split()
        except:
            return 0
        if 'day' in x[1]:
            return math.tanh(int(x[0]))
        elif 'week' in x[1]:
            return math.tanh(int(x[0]) * 7)
        elif 'month' in x[1]:
            return math.tanh(int(x[0]) * 30)
        elif 'year' in x[1]:
            return math.tanh(int(x[0]) * 365 + 1)

    test['Age'] = test['AgeuponOutcome'].map(Age)
    test.loc[(test['Age'] == 0), 'Age'] = test['Age'].mean()
    test['Age'] = test['Age'].astype(int)

    # sex
    def Type(string):
        try:
            x = string.split()
        except:
            return None
        if len(x) == 1:
            return None
        if x[0] == 'Intact':
            return True
        else:
            return False

    test['Type'] = test['SexuponOutcome'].map(Type)

    def Gender(string):
        try:
            x = string.split()
        except:
            return None
        if len(x) == 1:
            return None
        elif x[1] == 'Male':
            return True
        else:
            return False

    test['Gender'] = test['SexuponOutcome'].map(Gender)

    # name
    def isName(name):
        try:
            x = len(name)
        except:
            return False
        return True

    test['isName'] = test['Name'].map(isName)

    # shades
    def isShade(color):
        shades = ["Merle", "Brindle", "Tiger", "Smoke", "Cream", "Point", "Tick", "Tabby"]
        try:
            x = color.split('/')
        except:
            y = color.split()
            if len(y) == 1:
                return False
            else:
                if y[-1] in shades:
                    return True
                else:
                    return False
        for item in x:
            y = item.split()
            if y[-1] in shades:
                return True
            else:
                return False

    test['isShade'] = test['Color'].map(isShade)

    shades = ["Merle", "Brindle", "Tiger", "Smoke", "Cream", "Point", "Tick", "Tabby"]
    def Shade(color):
        try:
            x = color.split('/')
        except:
            y = color.split()
            if len(y) == 1:
                return False
            else:
                if shade == y[-1]:
                    return True
                else:
                    return False
        for item in x:
            y = item.split()
            if shade == y[-1]:
                return True
            else:
                return False

    for shade in shades:
        test[shade] = test['Color'].map(Shade)

    def isShade(color):
        try:
            x = color.split('/')
        except:
            y = color.split()
            if len(y) == 1:
                return False
            else:
                if y[-1] in shades:
                    return True
                else:
                    return False
        for item in x:
            y = item.split()
            if y[-1] in shades:
                return True
            else:
                return False

    for shade in shades:
        test["isShade"] = test['Color'].map(isShade)

    # colors
    colors = []
    a = train.Color.str.split('/')
    b = test.Color.str.split('/')
    ab = a.append(b)
    for i in ab:
        for c in i:
            colors.append(c)
    colors = set(colors)

    def isColor(color_s):
        try:
            x = color_s.split('/')
        except:
            if color == color_s:
                return True
            else:
                return False
        if color in x:
            return True
        else:
            return False

    for color in colors:
        test[color] = test['Color'].map(isColor)

    def Count(colors):
        try:
            x = colors.split('/')
        except:
            return 1
        if x:
            return len(x)

    test['NbColor'] = test['Color'].map(Count)

    # breeds
    breeds = []
    a = train.Breed.str.split('/')
    b = test.Breed.str.split('/')
    ab = a.append(b)
    for i in ab:
        for c in i:
            if c[-3:] == 'Mix':
                c = c[:-4]
                breeds.append(c)
            else:
                breeds.append(c)
    breeds = set(breeds)

    def isBreed(breed_s):
        if breed_s[-3:] == 'Mix':
            breed_s = breed_s[:-4]
            try:
                x = breed_s.split('/')
            except:
                if breed == breed_s:
                    return True
                else:
                    return False
            if breed in x:
                return True
            else:
                return False
        else:
            try:
                x = breed_s.split('/')
            except:
                if breed == breed_s:
                    return True
                else:
                    return False
            if breed in x:
                return True
            else:
                return False

    for breed in breeds:
        test[breed] = test['Breed'].map(isBreed)

    def Count_breed(breeds):
        try:
            x = breeds.split('/')
        except:
            return 1
        if x:
            return len(x)

    test['NbBreed'] = test['Breed'].map(Count_breed)

    def isMix(breeds):
        if breeds[-3:] == 'Mix':
            return True
        else:
            return False

    test['isMix'] = test['Breed'].map(isMix)

    # color + breed + mix
    test['ColorBreed'] = test['NbColor'] + test['NbBreed'] + test['isMix'] - 2

    # drop irrelevant attributes
    test.drop(['ID', 'DateTime', 'Name'], axis=1, inplace=True)

    return test


if __name__ == "__main__":
    # engineer the data
    train = clean_train(train)
    test = clean_test(test)

    # drop irrelevant attributes
    train.drop(['Color', 'Breed', 'AgeuponOutcome', 'SexuponOutcome'], axis=1, inplace=True)
    test.drop(['Color', 'Breed', 'AgeuponOutcome', 'SexuponOutcome'], axis=1, inplace=True)

    # transform to matrix
    train = train.values
    test = test.values

    # split into training and test set for training the model
    X_train, X_test, y_train, y_test = train_test_split(train[0::, 1::], train[0::, 0], test_size=0.2)

    # create the model with tuned parameters
    xgb1 = XGBClassifier(
        learning_rate=0.2,
        n_estimators=400,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        reg_lambda=3)

    # fitting the model with the training set
    print 'Fitting...\n'
    xgb1.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="mlogloss", eval_set=[(X_test, y_test)])
    # xgb2.fit(train[0::, 1::], train[0::, 0])

    # make the predictions from the test set
    print 'Predicting...\n'
    predictions = xgb1.predict_proba(test)

    # generate the output with panda framework
    print 'Making the output...\n'
    output = pd.DataFrame(predictions, columns=['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'])
    output.index += 1

    print(output)

    # convert panda to csv
    output.to_csv('predictions.csv', index_label='ID')

    print 'Done :-)'