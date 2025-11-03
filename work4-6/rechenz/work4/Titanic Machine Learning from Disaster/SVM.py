import pandas as pd
from sklearn import svm


def preprocess_data():
    train_data = pd.read_csv('datasets/titanic/train.csv')
    test_data = pd.read_csv('datasets/titanic/test.csv')
    train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],
                    axis=1, inplace=True)
    test_id = test_data['PassengerId']
    test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],
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
    train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
    test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
    train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].mean())
    test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())
    return train_data, test_data, test_id


def train(train_data):
    model = svm.SVC()
    model.fit(train_data.drop('Survived', axis=1),
              train_data['Survived'])
    return model


def predict_and_submit(model, test_data, test_id):
    predictions = model.predict(test_data)

    submission = pd.DataFrame({
        "PassengerId": test_id,
        "Survived": predictions
    })
    submission.to_csv('datasets/titanic/submission.csv', index=False)
    print('submission.csv has been created.')


def main():
    train_data, test_data, test_id = preprocess_data()
    model = train(train_data)
    predict_and_submit(model, test_data, test_id)


if __name__ == '__main__':
    main()
