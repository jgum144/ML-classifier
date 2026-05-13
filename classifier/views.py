from django.shortcuts import render  # Import render to handle HTML template responses
from django.http import JsonResponse  # Import JsonResponse for API responses
from django.views.decorators.csrf import csrf_exempt  # Import decorator to bypass CSRF for testing
import pickle  # Import pickle to load the serialized model
import numpy as np  # Import numpy for numerical array operations
import json  # Import json to parse incoming request data
import os  # Import os for file path management
from django.conf import settings  # Import settings to access project-level configurations

# Construct the absolute path to the pre-trained flower model file
MODEL_PATH = os.path.join(settings.BASE_DIR, 'flower_model.pkl')

# Attempt to load the trained model, scaler, and metadata from the pickle file
try:
    with open(MODEL_PATH, 'rb') as f:  # Open the file in binary read mode
        data = pickle.load(f)  # Load the dictionary containing all model data
        model = data['model']  # Extract the trained KNeighborsClassifier
        scaler = data['scaler']  # Extract the fitted StandardScaler
        target_names = data['target_names']  # Extract the human-readable species names
        confusion_matrix_data = data.get('confusion_matrix', [])  # Extract CM data if available
except FileNotFoundError:  # Handle cases where the model file is missing
    model = None  # Initialize model as None if loading fails
    scaler = None  # Initialize scaler as None if loading fails
    target_names = []  # Initialize empty names list
    confusion_matrix_data = []  # Initialize empty matrix

def home(request):  # View function for the main landing page
    context = {  # Prepare the data to be used in the template
        'target_names': target_names,  # Pass species names for headers
        'confusion_matrix': confusion_matrix_data,  # Pass raw CM data for display logic
        'zipped_cm': zip(target_names, confusion_matrix_data) if confusion_matrix_data else []  # Group labels with data
    }
    return render(request, 'classifier/index.html', context)  # Render the UI with the performance data

@csrf_exempt  # Exempt this view from CSRF token verification (for easy AJAX testing)
def predict(request):  # API view function to handle flower classification requests
    if request.method == 'POST':  # Only allow HTTP POST requests
        try:
            if not model:  # Check if the model was loaded successfully
                return JsonResponse({'error': 'Model not found. Run train_and_save.py first.'}, status=500)
            
            data = json.loads(request.body)  # Parse the incoming JSON request body
            features = [  # Extract the 4 required iris measurements from the input data
                float(data['sepal_length']),  # Extract sepal length as float
                float(data['sepal_width']),  # Extract sepal width as float
                float(data['petal_length']),  # Extract petal length as float
                float(data['petal_width'])  # Extract petal width as float
            ]
            
            # Prepare the input features for the machine learning model
            input_data = np.array([features])  # Convert the list to a 2D numpy array
            input_scaled = scaler.transform(input_data)  # Apply the same scaling used during training
            
            # Execute the prediction logic using the trained model
            prediction = model.predict(input_scaled)[0]  # Get the predicted class index (0, 1, or 2)
            prediction_proba = model.predict_proba(input_scaled)[0]  # Get probability scores for each class
            
            result = {  # Construct the response dictionary
                'class': target_names[int(prediction)],  # Map index to species name (e.g., 'setosa')
                'confidence': float(max(prediction_proba)) * 100,  # Get the highest probability as a percentage
                'probabilities': {name: float(prob) * 100 for name, prob in zip(target_names, prediction_proba)}  # Detailed breakdowns
            }
            
            return JsonResponse(result)  # Return the prediction result as JSON
        except Exception as e:  # Catch any processing errors
            return JsonResponse({'error': str(e)}, status=400)  # Return the error message with status 400
    return JsonResponse({'error': 'Invalid request method'}, status=405)  # Return error for GET/PUT/etc.
