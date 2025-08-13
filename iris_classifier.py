import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# loading dataset
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("First 10 rows of dataset:")
print(df.head(10))

# preparing data
X = iris.data
y = iris.target

# splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# training
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

# evaluation of model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# visualization of pairplot - u will see that setosa is separated from the others
pairplot = sns.pairplot(df, hue="target", height=2)
pairplot.fig.suptitle("Iris Dataset Feature Relationships", y=1.02)
pairplot.savefig("iris_pairplot.png")  # Saves the plot

# visualization of decision tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Visualization")
plt.savefig("iris_decision_tree.png")  # Saves the tree
plt.show()


# interactive Prediction (CLOSE BOTH PLOTS TO CONTINUE TO THE INPUT PROMPT)
print("\n--- Flower Prediction ---")
sl,sw,pl,pw = map(float, input("Enter sepal length, sepal width, petal length and petal width (cm): ").split())

user_data = np.array([[sl, sw, pl, pw]])
prediction = model.predict(user_data)
print(f"Predicted Species: {iris.target_names[prediction[0]]}")