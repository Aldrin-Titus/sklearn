import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

data = pd.read_csv('../dataset/iris.csv')

# 1. Split the dataset into features (X) and target (y)
X = df.drop('variety', axis=1)  # Features: sepal.length, sepal.width, petal.length, petal.width
y = df['variety']  # Target: flower variety (setosa, versicolor, etc.)

# 2. Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale the features (important for algorithms like KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Initialize the KNN classifier and train the model
knn = KNeighborsClassifier(n_neighbors=3)  # n_neighbors = 3 is commonly used
knn.fit(X_train_scaled, y_train)

# 5. Make predictions on the test set
y_pred = knn.predict(X_test_scaled)

# 6. Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

#To display the average values
def display_averages(df):
    avg_sepal_length = df['sepal.length'].mean()
    avg_sepal_width = df['sepal.width'].mean()
    avg_petal_length = df['petal.length'].mean()
    avg_petal_width = df['petal.width'].mean()

    print("\nAverage Values in the Dataset:")
    print(f"Average Sepal Length: {avg_sepal_length:.2f} cm")
    print(f"Average Sepal Width: {avg_sepal_width:.2f} cm")
    print(f"Average Petal Length: {avg_petal_length:.2f} cm")
    print(f"Average Petal Width: {avg_petal_width:.2f} cm")

# 7. Allow the user to input flower measurements for prediction
def user_input():
    try:
        display_averages(df)
        
        sepal_length = float(input("Enter sepal length (cm): "))
        sepal_width = float(input("Enter sepal width (cm): "))
        petal_length = float(input("Enter petal length (cm): "))
        petal_width = float(input("Enter petal width (cm): "))
        
        # Create a dataframe for the input features
        user_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                 columns=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'])
        
        # Scale the input features using the same scaler
        user_data_scaled = scaler.transform(user_data)
        
        # Predict the flower variety
        prediction = knn.predict(user_data_scaled)
        
        print(f"The predicted flower variety is: {prediction[0]}")
    except ValueError:
        print("Invalid input. Please enter numerical values.")

# Call the user_input function to get input and predict
user_input()
