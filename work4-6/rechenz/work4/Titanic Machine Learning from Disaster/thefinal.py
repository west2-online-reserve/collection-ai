import pandas as pd
from sklearn import model_selection
import numpy as np


def exact_between(s, start, end):
    try:
        parts = s.split(start)
        if len(parts) < 2:
            return None
        middle = parts[1].split(end)
        if len(middle) < 1:
            return None
        return middle[0].strip()
    except:
        return None


def age_group(age):
    if age <= 12:
        return 'Child'
    elif age <= 18:
        return 'Teenager'
    elif age <= 30:
        return 'Young Adult'
    elif age <= 40:
        return 'Adult'
    elif age <= 50:
        return 'Middle Age'
    elif age <= 60:
        return 'Senior'
    else:
        return 'Old'


def extract_title(name):
    title = name.split(',')[1].split('.')[0].strip()
    title_dict = {
        'Mr': 'Mr',
        'Miss': 'Miss',
        'Mrs': 'Mrs',
        'Master': 'Master',
        'Dr': 'Rare',
        'Rev': 'Rare',
        'Major': 'Rare',
        'Col': 'Rare',
        'Mlle': 'Miss',
        'Mme': 'Mrs',
        'Ms': 'Miss',
        'Lady': 'Rare',
        'Sir': 'Rare',
        'Capt': 'Rare',
        'Countess': 'Rare',
        'Don': 'Rare',
        'Jonkheer': 'Rare'
    }
    return title_dict.get(title, 'Rare')


def preprocess_data():
    train_data = pd.read_csv('datasets/titanic/train.csv')
    test_data = pd.read_csv('datasets/titanic/test.csv')
    train_data.drop(['PassengerId', 'Ticket'],
                    axis=1, inplace=True)
    test_id = test_data['PassengerId']
    test_data.drop(['PassengerId', 'Ticket'],
                   axis=1, inplace=True)
    train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
    test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
    Embarked_encoding = pd.get_dummies(
        train_data['Embarked'], prefix='Embarked', drop_first=True)
    Em = pd.get_dummies(test_data['Embarked'],
                        prefix='Embarked', drop_first=True)
    train_data = pd.concat([train_data, Embarked_encoding], axis=1)
    test_data = pd.concat([test_data, Em], axis=1)
    train_data.drop(['Embarked'], axis=1, inplace=True)
    test_data.drop(['Embarked'], axis=1, inplace=True)
    train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
    test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
    train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].median())
    test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
    train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']+1
    test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']+1
    train_data['Cabin'].fillna(0, inplace=True)
    test_data['Cabin'].fillna(0, inplace=True)
    train_data['Cabin'] = train_data['Cabin'].map(lambda x: 0 if x != 1 else 1)
    test_data['Cabin'] = test_data['Cabin'].map(lambda x: 0 if x != 1 else 1)
    train_data.drop(['SibSp', 'Parch'], axis=1, inplace=True)
    test_data.drop(['SibSp', 'Parch'], axis=1, inplace=True)
    train_data['AgeGroup'] = train_data['Age'].apply(age_group)
    test_data['AgeGroup'] = test_data['Age'].apply(age_group)
    # train_data.drop(['Age'], axis=1, inplace=True)
    # test_data.drop(['Age'], axis=1, inplace=True)
    # temp = pd.get_dummies(
    #     train_data['AgeGroup'], prefix='AgeGroup', drop_first=True)
    # train_data = pd.concat([train_data, temp], axis=1)
    # temp = pd.get_dummies(test_data['AgeGroup'],
    #                       prefix='AgeGroup', drop_first=True)
    # test_data = pd.concat([test_data, temp], axis=1)
    # train_data.drop(['AgeGroup'], axis=1, inplace=True)
    # test_data.drop(['AgeGroup'], axis=1, inplace=True)
    train_data['Title'] = train_data['Name'].apply(extract_title)
    test_data['Title'] = test_data['Name'].apply(extract_title)
    train_data.drop(['Name'], axis=1, inplace=True)
    test_data.drop(['Name'], axis=1, inplace=True)
    # 2. 家庭特征
    train_data['IsAlone'] = (train_data['FamilySize'] == 1).astype(int)
    test_data['IsAlone'] = (test_data['FamilySize'] == 1).astype(int)

    # 3. 年龄和性别的组合
    train_data['Sex_Age'] = train_data['Sex'].astype(
        str) + '_' + train_data['AgeGroup'].astype(str)
    test_data['Sex_Age'] = test_data['Sex'].astype(
        str) + '_' + test_data['AgeGroup'].astype(str)
    # train_data.drop(['Sex', 'AgeGroup'], axis=1, inplace=True)
    # test_data.drop(['Sex', 'AgeGroup'], axis=1, inplace=True)
    # 4. 船舱特征
    train_data['HasCabin'] = train_data['Cabin'].notna().astype(int)
    test_data['HasCabin'] = test_data['Cabin'].notna().astype(int)

    # 5. 票价分组
    train_data['FareGroup'] = pd.qcut(train_data['Fare'], 4, labels=[
                                      'Low', 'Medium', 'High', 'Very High'])
    test_data['FareGroup'] = pd.qcut(test_data['Fare'], 4, labels=[
                                     'Low', 'Medium', 'High', 'Very High'])
    # train_data.drop(['Fare'], axis=1, inplace=True)
    # test_data.drop(['Fare'], axis=1, inplace=True)
    # print(train_data.to_string())
    from sklearn.preprocessing import LabelEncoder

    # 为每个分类特征使用独立的编码器
    # 处理AgeGroup
    le_age = LabelEncoder()
    train_data['AgeGroup'] = le_age.fit_transform(train_data['AgeGroup'])
    test_data['AgeGroup'] = le_age.transform(test_data['AgeGroup'])

    # 处理Title
    le_title = LabelEncoder()
    train_data['Title'] = le_title.fit_transform(train_data['Title'])
    test_data['Title'] = le_title.transform(test_data['Title'])

    # 处理Sex_Age
    le_sex_age = LabelEncoder()
    train_data['Sex_Age'] = le_sex_age.fit_transform(train_data['Sex_Age'])
    test_data['Sex_Age'] = le_sex_age.transform(test_data['Sex_Age'])

    # 处理FareGroup
    le_fare = LabelEncoder()
    train_data['FareGroup'] = le_fare.fit_transform(train_data['FareGroup'])
    test_data['FareGroup'] = le_fare.transform(test_data['FareGroup'])

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    numeric_features = ['Age', 'Fare', 'FamilySize']
    train_data[numeric_features] = scaler.fit_transform(
        train_data[numeric_features])
    test_data[numeric_features] = scaler.transform(test_data[numeric_features])

    return train_data, test_data, test_id


def train(train_data):
    X = train_data.drop(['Survived'], axis=1)
    y = train_data['Survived']

    from sklearn.ensemble import VotingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn import model_selection

    rf = RandomForestClassifier()
    rf = model_selection.GridSearchCV(rf, {'n_estimators': [100, 200, 300],
                                           'max_depth': [4, 5, 6, 10],
                                           'min_samples_split': [2, 3, 4],
                                           'min_samples_leaf': [2, 3, 4]},

                                      cv=5, n_jobs=-1)

    gbc = GradientBoostingClassifier()
    gbc = model_selection.GridSearchCV(gbc, {'n_estimators': [100, 200, 300],
                                             'learning_rate': [0.1, 0.05, 0.01, 0.005],
                                             'max_depth': [4, 5, 6, 10],
                                             'min_samples_split': [2, 3, 4],
                                             'min_samples_leaf': [2, 3, 4]},

                                       cv=5, n_jobs=-1)

    lr = LogisticRegression()
    lr = model_selection.GridSearchCV(lr, {'max_iter': [100, 200, 300, 500, 1000],
                                           'C': [0.1, 1, 10, 100, 1000]},

                                      cv=5, n_jobs=-1)
    model = VotingClassifier(
        estimators=[('rf', rf), ('gbc', gbc), ('lr', lr)],
        voting='soft'
    )

    model.fit(X, y)

    return model


def predict_and_submit(model, test_data, test_id):
    predictions = model.predict(test_data)

    submission = pd.DataFrame({
        "PassengerId": test_id,
        "Survived": predictions
    })
    submission.to_csv('datasets/titanic/submission.csv', index=False)
    print('submission.csv has been created.')


def predict(model, test_data):
    ans = test_data['Survived']
    test_data.drop(['Survived'], axis=1, inplace=True)
    predictions = model.predict(test_data)
    acc = sum(predictions == ans)/len(predictions)
    print(acc)
    return predictions


def select_features(train_data, test_data):
    from sklearn.feature_selection import SelectKBest, f_classif

    X = train_data.drop(['Survived'], axis=1)
    y = train_data['Survived']

    # 确保没有NaN值
    if y.isnull().any():
        print("Found NaN values in target variable")
        train_data = train_data.dropna(subset=['Survived'])
        X = train_data.drop(['Survived'], axis=1)
        y = train_data['Survived']

    # 检查特征数量
    n_features = X.shape[1]
    k = min(11, n_features)  # 确保k不超过特征数量

    # 检查常数特征
    constant_features = X.columns[X.nunique() == 1]
    if len(constant_features) > 0:
        print(f"Removing constant features: {list(constant_features)}")
        X = X.drop(columns=constant_features)

    # 选择最重要的特征
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X, y)

    # 获取选中的特征名称
    selected_features = X.columns[selector.get_support()]

    # 更新训练集和测试集
    train_data = pd.concat(
        [pd.DataFrame(X_new, columns=selected_features), train_data['Survived']], axis=1)

    # 确保测试集只包含选中的特征
    test_data = test_data[selected_features]

    return train_data, test_data


def main():
    train_data, test_data, test_id = preprocess_data()
    train_data, test_data = select_features(train_data, test_data)
    traind, test = model_selection.train_test_split(
        train_data, test_size=0.1, random_state=42)
    # model = train(traind)
    # predict(model, test)
    # train_data, test_data = select_features(train_data, test_data)
    model = train(train_data)
    predict_and_submit(model, test_data, test_id)


if __name__ == '__main__':
    main()
