import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# pre-processing
def clean_train(train):
    # map animal types with integers (0 and 1)
    animal_type = {'Cat': 0, 'Dog': 1}
    train['AnimalType'] = train['AnimalType'].map(animal_type)

    # time of the day
    train['Hour'] = train['DateTime'].str[11:13].astype(int)
    train['Minute'] = train['DateTime'].str[14:16].astype(int)

    # create weekday attribute
    train['DateTime'] = pd.to_datetime(train['DateTime'])
    train['WeekDay'] = train['DateTime'].dt.dayofweek

    # create month attribute
    train['Month'] = train['DateTime'].dt.month

    # outcome
    outcome = {'Adoption': 0, 'Died': 1, 'Euthanasia': 2, 'Return_to_owner': 3, 'Transfer': 4}
    train['OutcomeType'] = train['OutcomeType'].map(outcome)

    # age
    def years(age):
        try:
            x = age.split()
        except:
            return 0
        if (x[1] == 'years' or x[1] == 'year'):
            return int(x[0])
        else:
            return 0

    train['AgeYears'] = train['AgeuponOutcome'].map(years)

    def months(age):
        try:
            x = age.split()
        except:
            return 0
        if (x[1] == 'months' or x[1] == 'month'):
            return x[0]
        else:
            return 0

    train['AgeMonths'] = train['AgeuponOutcome'].map(months).astype(int)

    def weeks(age):
        try:
            x = age.split()
        except:
            return 0
        if (x[1] == 'weeks' or x[1] == 'week'):
            return x[0]
        else:
            return 0

    train['AgeWeeks'] = train['AgeuponOutcome'].map(weeks).astype(int)

    def days(age):
        try:
            x = age.split()
        except:
            return 0
        if (x[1] == 'days' or x[1] == 'day'):
            return x[0]
        else:
            return 0

    train['AgeDays'] = train['AgeuponOutcome'].map(days).astype(int)

    def Age(age):
        try:
            x = age.split()
        except:
            return 0
        if 'day' in x[1]:
            return int(x[0])
        elif 'week' in x[1]:
            return int(x[0]) * 7
        elif 'month' in x[1]:
            return int(x[0]) * 30
        elif 'year' in x[1]:
            return int(x[0]) * 365

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
        if len(x)==1:
            return None
        elif x[1] == 'Male':
            return int(1)
        else:
            return int(0)

    train['Gender'] = train['SexuponOutcome'].map(Gender)


    # name
    def namelength(name):
        try:
            x = len(name)
        except:
            return 0
        return x

    train['NameLength'] = train['Name'].map(namelength)
    train.loc[(train['NameLength'].isnull()), 'NameLength'] = 0
    train['NameLength'] = train['NameLength'].astype(int)

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
                return 1
            else:
                return 0
        if color in x:
            return 1
        else:
            return 0

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
                    return 1
                else:
                    return 0
            if breed in x:
                return 1
            else:
                return 0
        else:
            try:
                x = breed_s.split('/')
            except:
                if breed == breed_s:
                    return 1
                else:
                    return 0
            if breed in x:
                return 1
            else:
                return 0

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
            return 1
        else:
            return 0

    train['isMix'] = train['Breed'].map(isMix)

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
    test['Hour'] = test['DateTime'].str[11:13].astype(int)
    test['Minute'] = test['DateTime'].str[14:16].astype(int)

    # create weekday attribute
    test['DateTime'] = pd.to_datetime(test['DateTime'])
    test['WeekDay'] = test['DateTime'].dt.dayofweek

    # age
    def years(age):
        try:
            x = age.split()
        except:
            return 0
        if (x[1] == 'years' or x[1] == 'year'):
            return int(x[0])
        else:
            return 0

    test['AgeYears'] = test['AgeuponOutcome'].map(years)

    def months(age):
        try:
            x = age.split()
        except:
            return 0
        if (x[1] == 'months' or x[1] == 'month'):
            return x[0]
        else:
            return 0

    test['AgeMonths'] = test['AgeuponOutcome'].map(months).astype(int)

    def weeks(age):
        try:
            x = age.split()
        except:
            return 0
        if (x[1] == 'weeks' or x[1] == 'week'):
            return x[0]
        else:
            return 0

    test['AgeWeeks'] = test['AgeuponOutcome'].map(weeks).astype(int)

    def days(age):
        try:
            x = age.split()
        except:
            return 0
        if (x[1] == 'days' or x[1] == 'day'):
            return x[0]
        else:
            return 0

    test['AgeDays'] = test['AgeuponOutcome'].map(days).astype(int)

    def Age(age):
        try:
            x = age.split()
        except:
            return 0
        if 'day' in x[1]:
            return int(x[0])
        elif 'week' in x[1]:
            return int(x[0]) * 7
        elif 'month' in x[1]:
            return int(x[0]) * 30
        elif 'year' in x[1]:
            return int(x[0]) * 365

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
            return int(1)
        else:
            return int(0)

    test['Type'] = test['SexuponOutcome'].map(Type)

    def Gender(string):
        try:
            x = string.split()
        except:
            return None
        if len(x) == 1:
            return None
        elif x[1] == 'Male':
            return int(1)
        else:
            return int(0)

    test['Gender'] = test['SexuponOutcome'].map(Gender)

    # name
    def namelength(name):
        try:
            x = len(name)
        except:
            return 0
        return x

    test['NameLength'] = test['Name'].map(namelength)
    test.loc[(train['NameLength'].isnull()), 'NameLength'] = 0
    test['NameLength'] = test['NameLength'].astype(int)

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
                return 1
            else:
                return 0
        if color in x:
            return 1
        else:
            return 0

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
                    return 1
                else:
                    return 0
            if breed in x:
                return 1
            else:
                return 0
        else:
            try:
                x = breed_s.split('/')
            except:
                if breed == breed_s:
                    return 1
                else:
                    return 0
            if breed in x:
                return 1
            else:
                return 0

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
            return 1
        else:
            return 0

    test['isMix'] = test['Breed'].map(isMix)

    # drop irrelevant attributes
    test.drop(['ID', 'DateTime', 'Name'], axis=1, inplace=True)

    return test


if __name__ == "__main__":
    train = clean_train(train)
    test = clean_test(test)

    train.drop(['Color', 'Breed', 'AgeuponOutcome'], axis=1, inplace=True)
    test.drop(['Color', 'Breed', 'AgeuponOutcome'], axis=1, inplace=True)

    # train = train.values
    # test = test.values

    # print train

    # print train.columns

    print train['Month']


    # X_train, X_test, y_train, y_test = train_test_split(train[0::, 1::], train[0::, 0], test_size=0.2)


    ### RFD ###

    # rf = RandomForestClassifier(n_estimators=500, max_features='auto')
    # print 'Fitting...\n'
    # rf.fit(X_train, y_train)
    # print 'Predicting...\n'
    # y_pred = rf.predict(X_test)
    # print accuracy_score(y_test, y_pred), 'of accuracy.\n'


    ## PREDICTIONS FOREST ##

    # print 'Fitting...\n'
    # forest = RandomForestClassifier(n_estimators = 500, max_features='auto')
    # forest = forest.fit(train[0::,1::],train[0::,0])
    # print("Predicting...\n")
    # predictions = forest.predict_proba(test)
    #
    # output = pd.DataFrame(predictions, columns=['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'])
    # output.index += 1
    #
    # print(output)
    #
    # output.to_csv('predictions.csv', index_label='ID')

    print 'Done.'


