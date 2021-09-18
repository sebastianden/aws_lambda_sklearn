import json
from joblib import load
from sklearn.ensemble import RandomForestClassifier

target_names = ['setosa', 'versicolor', 'virginica']

rfc = load('model.joblib')


def lambda_handler(event, context):

    if event["httpMethod"] == "POST":
        req = json.loads(event['body'])
    elif event["httpMethod"] == "GET":
        req = event["queryStringParameters"]

    sepal_length = float(req['sl'])
    sepal_width = float(req['sw'])
    petal_length = float(req['pl'])
    petal_width = float(req['pw'])

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
