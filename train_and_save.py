import numpy as np  # Import numpy for numeric array operations
import pickle  # Import pickle for model serialization
from sklearn.datasets import load_iris  # Import iris dataset loader
from sklearn.model_selection import train_test_split  # Import data splitter
from sklearn.preprocessing import StandardScaler  # Import feature scaler
from sklearn.neighbors import KNeighborsClassifier  # Import the KNN classifier
from sklearn.metrics import confusion_matrix  # Import confusion matrix metric


def train_and_save():  # Function to train and persist the model
    # Load the classic Iris flower dataset
    iris = load_iris()  # Load dataset into memory
    X = iris.data  # Assign the 4 feature columns to X
    y = iris.target  # Assign the target labels (species) to y

    # Split the dataset into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(  # 80/20 split
        X, y, test_size=0.20, random_state=42, stratify=y  # Ensure balanced classes in test set
    )

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()  # Initialize the scaler object
    X_train_scaled = scaler.fit_transform(X_train)  # Calculate mean/std and scale training data
    X_test_scaled = scaler.transform(X_test)  # Apply the same scaling to the test data

    # Initialize and train the K-Nearest Neighbors classifier
    model = KNeighborsClassifier(n_neighbors=5)  # Use 5 neighbors for voting
    model.fit(X_train_scaled, y_train)  # Train the model on scaled data

    # Evaluate the model's performance on the unseen test data
    y_pred = model.predict(X_test_scaled)  # Generate predictions for the test set
    cm = confusion_matrix(y_test, y_pred)  # Generate a confusion matrix

    # Create a dictionary to bundle all necessary components for prediction
    data = {  # Bundle model, scaler, and metadata
        "model": model,  # The trained classifier
        "scaler": scaler,  # The fitted scaler for new inputs
        "target_names": iris.target_names.tolist(),  # The names of the 3 iris species
        "confusion_matrix": cm.tolist(),  # The performance matrix for the UI
    }

    # Persist the bundled data to a file using pickle
    with open("flower_model.pkl", "wb") as f:  # Open file in write-binary mode
        pickle.dump(data, f)  # Serialize the data dictionary to the file

    # Print success message to the console
    print("Model, scaler, and confusion matrix saved to flower_model.pkl")


if __name__ == "__main__":  # Execute if the script is run directly
    train_and_save()  # Call the training function
