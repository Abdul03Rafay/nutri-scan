import React from 'react';
import { getNutritionInfo } from '../data/nutritionDB';

/**
 * A component to display nutritional information for a food item
 * 
 * @param {Object} props
 * @param {string} props.food - The food class name 
 * @returns {JSX.Element} - The rendered component
 */
const NutritionCard = ({ food }) => {
  const data = getNutritionInfo(food);
  
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

export default NutritionCard;