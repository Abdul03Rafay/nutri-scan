# Food Image Classification System + Nutrition Info.

### Abdul Rafay & Emily Yang

An end-to-end system for classifying food images and displaying nutritional information using TensorFlow and React.

<p align="center">
  <img src="https://github.com/user-attachments/assets/bb1746de-e92d-43ea-a01b-f8b913ff0312" width="400"/>
</p>


## Features

- Classify 101 different types of food from images
- Display detailed nutritional information for identified foods
- Easy-to-use interface with drag-and-drop image upload
- Runs entirely in the browser using TensorFlow.js (no server required)
- We were trying to deploy webinterfaceusing githb pages, but currently lacking model integration.

## Project Structure

The project consists of two main parts:

1. **Machine Learning (ML)**: Python code for training and converting the model # Our main focus
2. **Frontend**: React application for user interaction and displaying results # A good edition for replicability; incomplete as of now.



## Prerequisites

### For ML Training

- Python 3.7+
- TensorFlow 2.5+
- TensorFlow Datasets
- TensorFlow.js converter

### For Frontend

- Node.js 14+
- npm or yarn

## Installation

### Step 1: Set Up Python Environment for Training

```bash
# Clone the repository
git clone https://github.com/yourusername/food-classifier.git
cd food-classifier

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
cd ml
pip install -r requirements.txt
```

### Step 2: Train and Convert the Model

```bash
# Train the model (this will download the Food-101 dataset automatically)
python food_classifier.py

# Convert the model to TensorFlow.js format
python model_converter.py
```

### Step 3: Set Up the Frontend

```bash
# Navigate to the frontend directory
cd ../frontend

# Install dependencies
npm install

# Start the development server
npm start
```

## Using the Application

1. Open the application in your web browser
2. Upload an image of food using drag-and-drop or the file browser
3. Click the "Analyze Image" button
4. View the classification result and nutritional information

## How It Works

1. **Data Preparation**: The Food-101 dataset is loaded directly using TensorFlow Datasets
2. **Model Architecture**: Uses MobileNetV2 as a base model with custom classification layers
3. **Training Process**: Transfer learning with initial frozen layers, then fine-tuning
4. **Conversion**: The trained model is converted to TensorFlow.js format
5. **Frontend**: React application loads the model and performs classification in the browser
6. **Nutrition Data**: Predefined database maps food classes to nutritional information

## Extending the Project to Imrove on Current Limitations

### Adding More Food Classes

1. Use a different dataset or combine multiple datasets
2. Update the `nutritionDB.js` file with new food classes

### Improving Model Performance

1. Try different base models (EfficientNet, ResNet, etc.) # We also tried VGG16, which wasn't promising.
2. Use more sophisticated data augmentation techniques
3. Implement ensemble methods with multiple models

### Enhancing the Frontend

1. Have the trained model loaded properly

## Acknowledgments

- Food-101 dataset from ETH Zurich
- TensorFlow and TensorFlow.js teams
- USDA FoodData Central for nutrition information
