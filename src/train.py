from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump

# Import the Iris dataset
iris = datasets.load_iris()

# Split into data and target vector
X = iris['data']
y = iris['target']

# Split into train and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Make a Random Forest classifier and train it
rfc = RandomForestClassifier()
rfc = rfc.fit(X_train,y_train)

# Evaluate the testing accuracy
y_pred = rfc.predict(X_test)
print("Testing accuracy: ", accuracy_score(y_test,y_pred))

# Store the trained model as .joblib file
dump(rfc, './myLambdaFunction/model.joblib')