# nutri-scan

**Emily Yang, Abdul Rafay**

---

**Collect images of food items and train a neural network to recognize each and convey related nutritional information e.g. calories, % sugar.**

### **Methods:**

- **Model Architecture:** Employ a **CNN-based object detection mode**l to identify food objects.
- **Transfer Learning:** Experiment with pre-trained models; **ResNet**, **VGG16**, **EfficientNet, etc**
- **Data Augmentation:** Try to improve the model's robustness and generalization.
- **Nutritional Data Integration:** Post classification, food item will be **mapped to it’s nutritional facts**.
- Though unlikely, we may explore self-supervised or semi-supervised learning if labeled data is limited.

### **Data Sources:**

We plan to use from following publicly available datasets:

Image Classification

- **Google Open Images Dataset:** https://storage.googleapis.com/openimages/web/index.html — General
- **ImageNet:** https://www.image-net.org/ — General Image Library
- **Food-101** – 101 food categories, 101,000 images (1000 per category) https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
- **UECFOOD256** – 256 food categories, Japanese dishes, bounding boxes included
    
    https://huggingface.co/datasets/tiennv/uecfood256
    
- **VIREO Food-172** – 172 categories, 110k images, mostly Chinese dishes
    
    https://fvl.fudan.edu.cn/dataset/vireofood172/list.htm
    
- **Recipe1M+** – Food images with associated recipes, ingredients https://pic2recipe.csail.mit.edu/

Nutrition Facts Mapping

- **USDA FoodData Central** — offers detailed nutrient profiles for a wide range of foods, including branded and generic items. https://fdc.nal.usda.gov/ — API Available
- **Open Food Facts** — database of food products from around the world, providing information on ingredients, nutritional content. https://world.openfoodfacts.org/data — API Available

### **Code resources:**

**CNN Implementation Resources**

- https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
- https://pytorch.org/hub/pytorch_vision_resnet/

**Transfer Learning Tutorials**

- **PyTorch Transfer Learning Guide (Official)** https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- **TensorFlow Transfer Learning with EfficientNet** https://www.tensorflow.org/tutorials/images/transfer_learning

**Data Augmentation Libraries**

- **Albumentations**: Powerful image augmentation library with easy PyTorch and TensorFlow integration https://albumentations.ai/docs/

**Object Detection Frameworks**

- **Detectron2 (Facebook AI)**: High-performance object detection toolkit (for future expansion to object detection tasks like multiple foods per image) https://github.com/facebookresearch/detectron2
- **YOLOv5 (Ultralytics)**: Pre-trained object detection models, lightweight and fast https://github.com/ultralytics/yolov5

**Food Recognition End-to-End Examples for Reference**

- **Kaggle: Food-101 Classification Using CNN**
- **FastAI Food Classifier Tutorial**

### **What's new? — In addition to training the model to be able to classify food items, we will go further to provide relevant nutritional information as well.**

### **Project Implementation Timeline:**

| **Mar 25:** | Data collection and preprocessing completed |
| --- | --- |
| **April 10:** | Initial model training and baseline performance evaluation |
| **April 20:** | Experimentations with optimizations (data aug, model tuning) |
| **April 30:** | Final model evaluation, report writing, and presentation preparation |

### **Proposed demonstration/evaluation.**

**Classification Accuracy:** evaluate accuracy, precision, and recall on test datasets.

**Generalization Testing:** Test performance on unseen data for cross-dataset generalization.

**AI-Generated Image Evaluation:** Evaluate model’s performance on AI generated food images.

**Nutrition Fact Mapping Evaluation:** Post classification, evaluate the proposed nutrition fact with actual data.

## **Experiments.**

1.	**Model Architecture Variations**

- **Baseline CNN vs. Transfer Learning Models**: Compare a custom fine-tuned CNN model against pre-trained models like ResNet50, VGG16, and EfficientNet to evaluate transfer learning benefits.
- **Depth and Complexity Adjustments**: Vary the number of dense layers, convolutional layers, and dropout rates to measure overfitting, underfitting, and generalization impact.
- **Output Layer Extensions**: Test different ways to encode the nutritional information output, either as direct regression values or via lookup tables post-classification.

2.	**Data Augmentation Strategies**

Experiment with - **Rotation, flipping, scaling, color jitter, brightness adjustments, gaussian noise, blurring**

**Conduct ablation studies to evaluate performance changes when:**

- All augmentations applied

- Subsets applied

- No augmentation applied

**Measure classification accuracy and generalization robustness for each setup.**

3.	**Training Data Size Variation**

Evaluate model performance when trained on:

- Full dataset
- Reduced datasets (50%, 25% of total data) to simulate limited labeled data scenarios.
- Observe how training size affects accuracy and overfitting behavior.

### **Appendix.**

**Food Image Classification with Convolutional Neural Networks** by Malina Jiang from Stanford University

- https://cs230.stanford.edu/projects_fall_2019/reports/26233496.pdf

---

---

---

---

---

---

---

---

---

UPDATED: Regression CNN to predict nutrition information directly from the input image 

Dataset: https://github.com/google-research-datasets/Nutrition5k
