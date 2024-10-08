from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Load your dataset and initialize the scaler and model globally
data = pd.read_csv("cancer_data.csv")

# Assume 'diagnosis' is the column to predict
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split the data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a RandomForestClassifier (you can replace this with your neural network code)
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Test the model on the test set
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')

def visualize_feature_importance():
    # Get feature importances from the trained model
    importances = model.feature_importances_

    # Create a bar chart to visualize feature importances
    features = X.columns
    plt.bar(features, importances, color='blue')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')

    # Save the plot to a file
    plt.savefig('static/feature_importances.png')  # Save in the 'static' folder to serve it with Flask
    plt.close()

def visualize_overall_probability_chart():
    # Create an overall probability chart for the chosen dataset
    overall_probability = model.predict_proba(X_test_scaled)[:, 1] * 100

    # Create a bar chart to visualize overall probability
    plt.bar(range(len(overall_probability)), overall_probability, color='green')
    plt.xlabel('Data Points')
    plt.ylabel('Probability (%)')
    plt.title('Overall Probability Chart')

    # Save the plot to a file
    plt.savefig('static/overall_probability_chart.png')  # Save in the 'static' folder to serve it with Flask
    plt.close()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input values from the form
        data = request.form.to_dict()

        # Extract the relevant features and convert to DataFrame
        features = pd.DataFrame([data.values()], columns=data.keys())

        try:
            # Make sure the feature columns are the same as during training
            features = features[X.columns]
        except KeyError as e:
            # Log the error for debugging
            print(f"KeyError: {e}")
            return render_template('error.html', error_message="Invalid input features")

        # Standardize features using the same scaler
        features_scaled = scaler.transform(features)

        # Make predictions
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)[0][1] * 100

        result = 'Malignant' if prediction[0] == 1 else 'Benign'
        
        # Visualize feature importance
        visualize_feature_importance()

        # Visualize overall probability chart
        visualize_overall_probability_chart()

        return render_template('result.html', result=result, probability=probability, chart_image='static/feature_importances.png')

@app.route('/overall_chart')
def overall_chart():
    return render_template('overall_chart.html')

if __name__ == '__main__':
    app.run(debug=True)
