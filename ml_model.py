import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Step 1: Create the dataset
data = {
    "Id": list(range(1, 31)),
    "SepalLengthCm": [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9, 5.4, 4.8, 4.8, 4.3, 5.8, 5.7, 5.1, 5.7, 5.2, 5.0, 6.0, 5.5, 6.5, 5.7, 6.3, 5.8, 6.1, 6.4, 6.6, 6.7],
    "SepalWidthCm": [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4, 3.0, 3.0, 4.0, 3.8, 3.5, 3.8, 3.4, 3.2, 3.0, 2.4, 3.0, 2.8, 3.3, 2.7, 2.8, 3.2, 2.9, 3.1],
    "PetalLengthCm": [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 1.4, 1.1, 1.2, 1.7, 1.4, 1.7, 1.5, 1.2, 4.5, 4.0, 5.0, 4.2, 5.6, 5.1, 5.9, 5.5, 5.4, 5.7],
    "PetalWidthCm": [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.3, 0.3, 0.3, 0.2, 0.2, 1.5, 1.3, 1.8, 1.5, 2.1, 1.9, 2.3, 2.2, 2.0, 2.4],
    "Species": ["Iris-setosa"] * 20 + ["Iris-versicolor"] * 5 + ["Iris-virginica"] * 5
}

df = pd.DataFrame(data)
print("Dataset Loaded:\n", df.head())

# Step 2: Data Preprocessing
# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Normalize numerical features
scaler = MinMaxScaler()
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
df[features] = scaler.fit_transform(df[features])
print("\nNormalized Data:\n", df.head())

# Step 3: Model Selection and Training
X = df[features]  # Features
y = df["Species"]  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning for Decision Tree
param_grid = {
    'max_depth': [None, 3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best Model
model = grid_search.best_estimator_
print("\nBest Parameters:", grid_search.best_params_)

# Step 4: Model Evaluation
# Predict on test set
y_pred = model.predict(X_test)

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix with Heatmap
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Add sample details for confusion matrix visualization
results = pd.DataFrame({
    "Id": df.loc[X_test.index, "Id"],
    "SepalLengthCm": df.loc[X_test.index, "SepalLengthCm"],
    "SepalWidthCm": df.loc[X_test.index, "SepalWidthCm"],
    "PetalLengthCm": df.loc[X_test.index, "PetalLengthCm"],
    "PetalWidthCm": df.loc[X_test.index, "PetalWidthCm"],
    "Actual": y_test,
    "Predicted": y_pred
})

print("\nDetailed Confusion Matrix Results:\n")
print(results)

# Save the model
joblib.dump(model, "decision_tree_model.pkl")
print("\nModel saved as decision_tree_model.pkl")

# Step 5: Make Predictions
new_data = pd.DataFrame({
    "SepalLengthCm": [5.1, 6.3, 6.7],
    "SepalWidthCm": [3.5, 3.3, 3.1],
    "PetalLengthCm": [1.4, 5.6, 5.4],
    "PetalWidthCm": [0.2, 2.1, 2.0]
})

new_data[features] = scaler.transform(new_data[features])  # Normalize
predictions = model.predict(new_data)
print("\nPredictions for new data:\n", predictions)

# Step 6: Interactive Dashboard (Streamlit)
def interactive_dashboard():
    st.title("Iris Species Prediction")

    sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, step=0.1)
    sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, step=0.1)
    petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, step=0.1)
    petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, step=0.1)

    if st.button("Predict"):
        input_data = pd.DataFrame({
            "SepalLengthCm": [sepal_length],
            "SepalWidthCm": [sepal_width],
            "PetalLengthCm": [petal_length],
            "PetalWidthCm": [petal_width]
        })
        input_data[features] = scaler.transform(input_data[features])
        prediction = model.predict(input_data)
        st.write(f"Predicted Species: {prediction[0]}")

# Uncomment below to run Streamlit app
interactive_dashboard()  # Uncomment to run the Streamlit app
