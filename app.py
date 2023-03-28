
import streamlit as st
import pickle
import numpy as np



# Load the model
with open('Gauss.pickle', 'rb') as f:
    model = pickle.load(f)

# Create a function to get user input and make predictions
def predict_iris_class(sepal_length, sepal_width, petal_length, petal_width, k=3):
    # Convert user input to a numpy array
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Get the predicted class labels and their probabilities
    predicted_probabilities = model.predict_proba(input_features)[0]
    predicted_classes = model.classes_

    # Sort the predicted probabilities in descending order and get the top k classes and probabilities
    top_k_indices = predicted_probabilities.argsort()[::-1][:k]
    top_k_classes = predicted_classes[top_k_indices]
    top_k_probabilities = predicted_probabilities[top_k_indices]

    return top_k_classes, top_k_probabilities


# Create the Streamlit app
def main():
    # Set app title
    st.title("Iris Flower Classifier")

    # Add input fields for user to enter feature values
    sepal_length = st.slider('Sepal length', 0.0, 10.0, 5.0, step=0.1)
    sepal_width = st.slider('Sepal width', 0.0, 10.0, 5.0, step=0.1)
    petal_length = st.slider('Petal length', 0.0, 10.0, 5.0, step=0.1)
    petal_width = st.slider('Petal width', 0.0, 10.0, 5.0, step=0.1)

# When user clicks the 'Predict' button, make a prediction and display the result
    if st.button('Predict'):
        top_k_classes, top_k_probabilities = predict_iris_class(sepal_length, sepal_width, petal_length, petal_width, k=3)
        for i in range(3):
            st.write(f"{i+1}. {top_k_classes[i]} - Probability: {top_k_probabilities[i]:.2f}")


if __name__ == '__main__':
    main()