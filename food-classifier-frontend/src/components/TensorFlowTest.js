import React, { useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

const TensorFlowTest = () => {
  const [tfStatus, setTfStatus] = useState('Loading TensorFlow.js...');
  const [modelStatus, setModelStatus] = useState('');
  
  useEffect(() => {
    async function testTensorFlow() {
      try {
        // Test if TensorFlow.js works
        await tf.ready();
        setTfStatus('TensorFlow.js loaded successfully ✅');
        
        // Create a simple tensor to verify operations work
        const tensor = tf.tensor2d([[1, 2], [3, 4]]);
        const result = tensor.add(tf.scalar(1)).arraySync();
        console.log('Tensor operation result:', result);
        
        setModelStatus('Testing model loading...');
        
        // Try to load the model
        try {
          const model = await tf.loadLayersModel('/models/food_classifier_tfjs/model.json');
          console.log('Model loaded successfully');
          console.log('Model input shape:', model.inputs[0].shape);
          console.log('Model output shape:', model.outputs[0].shape);
          setModelStatus('Model loaded successfully ✅');
        } catch (modelError) {
          console.error('Error loading model:', modelError);
          setModelStatus(`Error loading model: ${modelError.message} ❌`);
        }
      } catch (error) {
        console.error('Error initializing TensorFlow:', error);
        setTfStatus(`Error initializing TensorFlow: ${error.message} ❌`);
      }
    }
    
    testTensorFlow();
  }, []);
  
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">TensorFlow.js Test</h1>
      
      <div className="bg-gray-100 p-4 rounded">
        <p className="font-bold">TensorFlow Status:</p>
        <p>{tfStatus}</p>
      </div>
      
      <div className="bg-gray-100 p-4 rounded mt-4">
        <p className="font-bold">Model Loading Status:</p>
        <p>{modelStatus}</p>
      </div>
      
      <div className="mt-4">
        <p>Check the browser console (F12) for more detailed log messages.</p>
      </div>
    </div>
  );
};

export default TensorFlowTest;