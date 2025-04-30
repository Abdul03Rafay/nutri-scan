import React, { useState } from 'react';
import './App.css';
import FoodClassifier from './components/FoodClassifier';
import TensorFlowTest from './components/TensorFlowTest';

function App() {
  const [showTest, setShowTest] = useState(true);
  
  return (
    <div className="App">
      <header className="bg-blue-600 text-white p-4 shadow-md">
        <h1 className="text-2xl font-bold">Food Classification & Nutrition App</h1>
        <p className="text-sm">Powered by machine learning and TensorFlow.js</p>
        
        <button 
          className="mt-2 px-3 py-1 bg-white text-blue-600 rounded"
          onClick={() => setShowTest(!showTest)}
        >
          {showTest ? 'Show Food Classifier' : 'Show TensorFlow Test'}
        </button>
      </header>
      
      <main className="container mx-auto px-4 py-8">
        {showTest ? (
          <TensorFlowTest />
        ) : (
          <FoodClassifier />
        )}
      </main>
      
      <footer className="bg-gray-100 p-4 text-center text-gray-600 text-sm mt-auto">
        <p>Food classification model trained on the Food-101 dataset</p>
        <p>© {new Date().getFullYear()} - Food Classification Project</p>
      </footer>
    </div>
  );
}

export default App;