# Serving a Machine Learning Model with AWS Lambda

This guide explains how to serve a [sci-kit learn](https://scikit-learn.org/stable/) ML model via 
[AWS Lambda](https://aws.amazon.com/lambda/) and [AWS API Gateway](https://aws.amazon.com/api-gateway/). Although the example is quite specific and the ML model simple, the method can be easily adopted for more difficult models. The concept of building an AWL Lambda layer can also be extended to other python package dependencies.

<div align="center">
	<img width=500 src="images/architecture.png" alt="architecture">
	<br>
    Application architecture with AWS API Gateway and AWS Lambda.
    <br>
    <br>
</div>

In the completed set-up a user is able to send HTTP requests to API Gateway triggering the execution of a Lambda function. This Lambda function will load the saved pre-trained model, load the dependencies from the right layer and make a prediction based on the input parameters provided by the user. The user recieves the result as request response.

## Contents

- [0. Prerequisites](README.md#0.-prerequisites)
- [1. Setting up an AWS Lambda layer](README.md#1.-setting-up-an-aws-lambda-layer)
- [2. Training a simple ML model](README.md#2.-training-a-simple-ml-model)
- [3. Setting up the Lambda function](README.md#3.-setting-up-the-lambda-function)
- [4. Configuring API Gateway](README.md#4.-configuring-api-gateway)


## 0. Prerequisites

Not many things are needed in order to get started:

- An AWS Account
- Local installation of [Python](https://www.python.org/) and [pip](https://pip.pypa.io/en/stable/) (to download the python packages)

## 1. Setting up an AWS Lambda layer
Lambda layers are a way of providing your Lambda function with the dependencies it needs to execute. In our case that would be the `scikit-learn` package, which in turn depends on a bunch of other packages.
Luckily we have pip to figure out the dependency tree for us. The only thing we have to do is specify the right version of the packages. There is a potential pitfall in this step (especially if you're working on a Windows machine) as you might think that you can simply reuse the packages already installed on your machine. However, AWS Lambda is running in a Linux environment, so the Python packages for Windows __won't work__! Instead we can ask pip to download the Linux versions of the dependencies we want by specifying `--platform manylinux1_x86_64`.  

```
pip download --python-version 38 --abi cp38 --platform manylinux1_x86_64 --only-binary=:all: --no-binary=:none: scikit-learn
```
Note: This example assumes our Lamda function is running on Pyton 3.8. If that is not the case simple adjust the `--python-version` ans `--abi` arguments. Pip will download the package `.whl` files to the location you are currently at. By specifying `--only-binary=:all: --no-binary=:none:` we tell pip that we also want to download all the packages that the `sklearn` package depends on.

Now we're nearly ready to make a Lambda layer. The only things we have to do now is unpack the `.whl` files and put them into the right folder structure. Unpacking is easy. We can do that on a Linux machine by calling `unzip path/to/file.whl` and on Windows by renaming `.whl` to `.zip` and simply extracting the files. Repeat this step for each package (in our case it should be `joblib`, `numpy`, `scikit_learn`, `scipy` and `threadpoolctl`). All folders called `*.dist-info` can safely be deleted. 

In order to make our Lambda function aware of the provided packages they have to be organized into a specific folder structure. The following diagram shows the structure and where to place all the extracted packages:

```
python/
└── lib/ 
    └── python3.8/
        └── site-packages/
            ├── joblib/  
            ├── numpy/
            ├── numpy.libs/
            ├── scikit_learn.libs/
            ├── scipy/
            ├── scipy.libs/
            ├── sklearn/
            └── threadpoolctl.py                    
```
Note: Again we are assuming that we are using Python 3.8. If you are using a different version adjust the folder name. Zip the whole folder structure before going on to the next step.

In the AWS Management Console search for "AWS Lambda". In the left-hand menu under "Additional resources" choose "Layers" and click "Create layer". Give the layer a name, upload the `.zip` file and choose a runtime (in our case Python 3.8). Note: For large files consider storing the `.zip` file in an S3 Bucket first and uploading it from there. Click "create".

<div align="center">
	<img width=500 src="images/layer1.PNG" alt="layer1">
    <br>
</div>

Congrats! You successfully created a Lambda layer!

## 2. Training a simple ML model

Now to some Data Science. Before we can serve a ML model and do inference, we have to create and train it. This toy example will train a Random Forest classifier on the [Iris data set](https://en.wikipedia.org/wiki/Iris_flower_data_set) and save the pretrained model as a file.
To view the whole code go to [train.py](src/train.py).

First we import some dependencies and load the dataset from the preinstalled datasets in `sklearn`. 
```python
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump

# Import the Iris dataset
iris = datasets.load_iris()
```
We split the data into training and testing sets ...
```python
# Split into data and target vector
X = iris['data']
y = iris['target']

# Split into train and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```
... and train a Random Forest Classifier on the training set.
```python
# Make a Random Forest classifier and train it
rfc = RandomForestClassifier()
rfc = rfc.fit(X_train,y_train)
```
We can validate the models performace by checking the testing accuracy:
```python
# Evaluate the testing accuracy
y_pred = rfc.predict(X_test)
print("Testing accuracy: ", accuracy_score(y_test,y_pred))
```
The output should be:
```
Testing accuracy:  0.98
```
In a last step we save the trained model as a `.joblib` file to preserve it and load it in our Lambda function.
```python
# Store the trained model as .joblib file
dump(rfc, './myLambdaFunction/model.joblib')
```

## 3. Setting up the Lambda function
Now we have nearly all pieces of the puzzle. The only thing missing is the Lambda function itself. In our case this will be another Python file containing a function following a special syntax. The so-called Lambda Handler. Each time the Lambda function is triggerd this function is executed and is provided input via the `event` variable. 
```python
def lambda_handler(event, context):
    # Load the features from the event dict
    sepal_length = float(event['sl'])
    sepal_width = float(event['sw'])
    petal_length = float(event['pl'])
    petal_width = float(event['pw'])
    # Set class names
    target_names = ['setosa','versicolor','virginica']
    # Load the pre-trained model
    rfc = load('model.joblib')
    # Predict a class based on the input
    y = rfc.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    # Turn encoding into class string
    c = target_names[y]
    print(c)
    # Return result as JSON
    return {
        'statusCode': 200,
        'body': c
    }
```
You can view the whole function in [lambda_function.py](src/myLambdaFunction/lambda_function.py). Zip "lambda_function.py" and model.joblib" before going on to the next step.

In the AWS Management Console search for "AWS Lambda". In the left-hand menu choose "Functions" and click "Create function". Give the function a name and select a runtime.

<div align="center">
	<img width=800 src="images/function1.PNG" alt="layer1">
    <br>
</div>

Under "Function code" click the "Action" drop-down and choose "Upload a .zip file". 

<div align="center">
	<img width=800 src="images/function2.PNG" alt="layer1">
    <br>
</div>

Upload the `.zip`file containing "lambda_function.py" and model.joblib". 

You can test the function by creating a Test. Go to Test > Configure Events > Create new test event. Provide a test name and the following body:
```JSON
{
  "sl": 6.9,
  "sw": 3.1,
  "pl": 5.1,
  "pw": 2.3
}
```
If you run the test the execution result should read:
```JSON
Response:
{
  "statusCode": 200,
  "body": "virginica"
}
```

## 4. Configuring API Gateway

This is great!. Our Lambda function works and is __hosting a ML model__! However, it has no way to communicate with the outside world and is pretty useless. Thus we need to define a trigger to activate the execution of the function and figure a way to input data.