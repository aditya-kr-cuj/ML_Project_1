# ML_Project_1
# **Iris Species Prediction**

This repository contains a machine learning project for predicting the species of an Iris flower based on its physical features (sepal and petal dimensions). The project uses a Decision Tree Classifier and includes a Streamlit-based interactive dashboard for real-time predictions.

## **Features**

- Preprocessing of the Iris dataset, including normalization.
- Decision Tree Classifier with hyperparameter tuning using GridSearchCV.
- Model evaluation through accuracy score, classification report, and confusion matrix visualization.
- Streamlit dashboard for interactive predictions.

## **Installation**

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>

## **Usage**
Run the Model Training Script

-To train the model and save it as a file:
-python ml_model.py

Run the Streamlit App

-To launch the interactive dashboard:
-streamlit run ml_model.py

## **Dependencies**
This project requires the following Python libraries:

pandas
scikit-learn
matplotlib
seaborn
joblib
streamlit
   
## **Project Structure**

|-- ml_model.py             # Main project script
|-- decision_tree_model.pkl # Saved model file
|-- requirements.txt        # Python dependencies
|-- README.md               # Project documentation
