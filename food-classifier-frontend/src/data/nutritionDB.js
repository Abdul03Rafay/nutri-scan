import React, { useState, useRef, useEffect } from 'react';
// Import TensorFlow.js as a single module
import * as tf from '@tensorflow/tfjs';

/**
 * Main component for food image classification
 * Handles image upload, model loading, and food classification
 */
const FoodClassifier = () => {
  // State for handling the application
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [model, setModel] = useState(null);
  const [classLabels, setClassLabels] = useState(null);
  const [modelLoading, setModelLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState(null);
  const fileInputRef = useRef(null);
  
  // Load the TensorFlow.js model and class labels
  useEffect(() => {
    async function loadModel() {
      try {
        setModelLoading(true);
        console.log("Loading TensorFlow.js model...");
        
        // Load the model
        const loadedModel = await tf.loadLayersModel('/models/food_classifier_tfjs/model.json');
        console.log("Model loaded successfully");
        setModel(loadedModel);
        
        // Load the class labels
        console.log("Loading class labels...");
        const response = await fetch('/models/class_indices.json');
        const labels = await response.json();
        setClassLabels(labels);
        console.log(`Loaded ${Object.keys(labels).length} class labels`);
        
        setModelLoading(false);
      } catch (error) {
        console.error('Error loading model:', error);
        setErrorMessage('Failed to load the food classification model. Please check if model files are correctly placed in the public/models directory.');
        setModelLoading(false);
      }
    }
    
    loadModel();
  }, []);
  
  // Format food name for display
  const formatFoodName = (name) => {
    if (!name) return '';
    
    // Replace underscores with spaces
    let formatted = name.replace(/_/g, ' ');
    
    // Capitalize first letter of each word
    return formatted
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };
  
  const preprocessImage = async (imgElement) => {
    return tf.tidy(() => {
      // Convert the image to a tensor
      let tensor = tf.browser.fromPixels(imgElement);
      
      // Resize the image to match the model's expected input dimensions (224x224)
      tensor = tf.image.resizeBilinear(tensor, [224, 224]);
      
      // Normalize pixel values to [0, 1]
      tensor = tensor.div(255.0);
      
      // Expand dimensions to create a batch of size 1
      tensor = tensor.expandDims(0);
      
      return tensor;
    });
  };
  
  const classifyImage = async (imgElement) => {
    try {
      // Preprocess the image
      const tensor = await preprocessImage(imgElement);
      
      // Make a prediction with the model
      const predictions = await model.predict(tensor);
      
      // Get the prediction probabilities
      const probabilities = await predictions.data();
      
      // Find the class with the highest probability
      const maxProbabilityIndex = probabilities.indexOf(Math.max(...probabilities));
      const predictedClass = classLabels[maxProbabilityIndex];
      
      // Cleanup tensor to prevent memory leaks
      tf.dispose(tensor);
      tf.dispose(predictions);
      
      return predictedClass;
    } catch (error) {
      console.error('Error during classification:', error);
      throw new Error('Failed to classify the image. Please try again.');
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = () => {
        setImage(reader.result);
        setPrediction(null);
        setErrorMessage(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = () => {
        setImage(reader.result);
        setPrediction(null);
        setErrorMessage(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const analyzeImage = async () => {
    if (!image || !model || !classLabels) return;
    
    setIsAnalyzing(true);
    setErrorMessage(null);
    
    try {
      // Create an image element for TensorFlow.js to use
      const imgElement = new Image();
      imgElement.src = image;
      
      // Wait for the image to load
      await new Promise((resolve) => {
        imgElement.onload = resolve;
      });
      
      // Classify the image
      const result = await classifyImage(imgElement);
      setPrediction(result);
    } catch (error) {
      console.error("Error analyzing image:", error);
      setErrorMessage(error.message || "An error occurred during analysis.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetAll = () => {
    setImage(null);
    setPrediction(null);
    setErrorMessage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Simplified nutrition data for the predicted food
  const getNutritionInfo = (food) => {
    // Sample nutrition database (in a real app, this would be more comprehensive)
    const nutritionDB = {
      pizza: {
        calories: 285,
        protein: "12g", 
        carbs: "36g",
        fiber: "2.5g",
        sugar: "3.8g", 
        fat: "10g",
        vitamins: ["Vitamin A", "Vitamin B12"],
        minerals: ["Calcium", "Iron"]
      },
      hamburger: {
        calories: 295,
        protein: "17g",
        carbs: "24g", 
        fiber: "1.3g",
        sugar: "4.4g",
        fat: "14g",
        vitamins: ["Vitamin B6", "Vitamin B12"],
        minerals: ["Iron", "Zinc", "Potassium"]
      },
      // Add more foods as needed
      apple_pie: {
        calories: 237,
        protein: "2.4g",
        carbs: "34g",
        fiber: "1.4g", 
        sugar: "18g",
        fat: "11g",
        vitamins: ["Vitamin A", "Vitamin C"],
        minerals: ["Iron", "Calcium"]
      }
    };
    
    // Return nutrition info for the food or default values if not found
    return nutritionDB[food] || {
      calories: "N/A",
      protein: "N/A",
      carbs: "N/A",
      fiber: "N/A",
      sugar: "N/A",
      fat: "N/A",
      vitamins: ["Information not available"],
      minerals: ["Information not available"]
    };
  };

  // Nutrition Card Component
  const NutritionCard = ({ food }) => {
    const data = getNutritionInfo(food);
    
    return (
      <div className="bg-white rounded-lg shadow-lg p-6 mt-6 w-full max-w-md">
        <div className="border-b-2 border-gray-800 pb-2 mb-4">
          <h3 className="text-2xl font-bold text-center">{formatFoodName(food)} Nutrition Facts</h3>
          <p className="text-center text-gray-600">Serving size: 100g</p>
        </div>
        
        <div className="border-b border-gray-300 pb-2 mb-2">
          <div className="flex justify-between">
            <span className="font-bold">Calories</span>
            <span>{data.calories}</span>
          </div>
        </div>
        
        <div className="border-b border-gray-300 pb-2 mb-2">
          <div className="flex justify-between">
            <span className="font-bold">Total Fat</span>
            <span>{data.fat}</span>
          </div>
        </div>
        
        <div className="border-b border-gray-300 pb-2 mb-2">
          <div className="flex justify-between">
            <span className="font-bold">Total Carbohydrates</span>
            <span>{data.carbs}</span>
          </div>
          <div className="flex justify-between pl-4">
            <span>Dietary Fiber</span>
            <span>{data.fiber}</span>
          </div>
          <div className="flex justify-between pl-4">
            <span>Sugar</span>
            <span>{data.sugar}</span>
          </div>
        </div>
        
        <div className="border-b border-gray-300 pb-2 mb-2">
          <div className="flex justify-between">
            <span className="font-bold">Protein</span>
            <span>{data.protein}</span>
          </div>
        </div>
        
        <div className="mb-2">
          <p className="font-bold">Vitamins:</p>
          <div className="flex flex-wrap gap-1">
            {data.vitamins.map((vitamin, index) => (
              <span key={index} className="bg-green-100 text-green-800 px-2 py-1 rounded text-sm">
                {vitamin}
              </span>
            ))}
          </div>
        </div>
        
        <div>
          <p className="font-bold">Minerals:</p>
          <div className="flex flex-wrap gap-1">
            {data.minerals.map((mineral, index) => (
              <span key={index} className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm">
                {mineral}
              </span>
            ))}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col items-center p-6 bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-bold mb-2 text-center">Food Image Classifier</h1>
      
      {modelLoading ? (
        <div className="text-center p-6">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500 mb-2"></div>
          <p>Loading food classification model...</p>
        </div>
      ) : (
        <>
          <p className="text-center text-gray-600 mb-6 max-w-lg">
            Upload an image of food to identify it and see its nutritional information.
          </p>
          
          {/* Image Upload Area */}
          <div 
            className="w-full max-w-lg bg-white rounded-lg border-2 border-dashed border-gray-300 p-6 mb-6 text-center"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            {!image ? (
              <div>
                <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                  <path 
                    d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" 
                    strokeWidth="2" 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                  />
                </svg>
                <p className="mt-2 text-sm text-gray-600">
                  Drag and drop an image here, or 
                  <label className="mx-1 text-blue-600 hover:text-blue-800 cursor-pointer">
                    browse
                    <input 
                      type="file" 
                      className="hidden" 
                      accept="image/*" 
                      onChange={handleFileChange}
                      ref={fileInputRef}
                    />
                  </label>
                </p>
                <p className="text-xs text-gray-500 mt-1">PNG, JPG, GIF up to 10MB</p>
              </div>
            ) : (
              <div className="relative">
                <img 
                  src={image} 
                  alt="Food preview" 
                  className="max-h-64 mx-auto rounded" 
                />
                <button 
                  onClick={resetAll}
                  className="absolute top-0 right-0 bg-red-500 text-white rounded-full p-1 transform translate-x-1/2 -translate-y-1/2"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </button>
              </div>
            )}
          </div>
          
          {/* Error Message */}
          {errorMessage && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4 max-w-lg w-full">
              <span className="block sm:inline">{errorMessage}</span>
            </div>
          )}
          
          {/* Analyze Button */}
          {image && !prediction && !errorMessage && (
            <button
              onClick={analyzeImage}
              disabled={isAnalyzing}
              className={`px-4 py-2 rounded font-medium ${
                isAnalyzing 
                  ? 'bg-gray-300 cursor-not-allowed' 
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              }`}
            >
              {isAnalyzing ? (
                <>
                  <span className="inline-block animate-spin rounded-full h-4 w-4 border-t-2 border-b-2 border-white mr-2"></span>
                  Analyzing...
                </>
              ) : 'Analyze Image'}
            </button>
          )}
          
          {/* Results */}
          {prediction && (
            <div className="w-full max-w-md">
              <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded relative mb-6">
                <strong className="font-bold">Food identified: </strong>
                <span className="block sm:inline">{formatFoodName(prediction)}</span>
              </div>
              
              <NutritionCard food={prediction} />
              
              <button
                onClick={resetAll}
                className="mt-6 px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 w-full"
              >
                Analyze Another Image
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default FoodClassifier;
