import json
from joblib import load
from sklearn.ensemble import RandomForestClassifier

target_names = ['setosa', 'versicolor', 'virginica']

rfc = load('model.joblib')


def lambda_handler(event, context):

    event = json.loads(event['body'])

    sepal_length = float(event['sl'])
    sepal_width = float(event['sw'])
    petal_length = float(event['pl'])
    petal_width = float(event['pw'])

    y = rfc.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    c = target_names[y]
    print(c)

    return {
        'statusCode': 200,
        'body': c,
        'headers': {
            'Content-Type': 'application/json',
        },
    }
