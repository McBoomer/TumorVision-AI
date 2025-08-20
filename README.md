# Brain Tumor MRI Classifier

### By Muadh Khan  
*Markham, Ontario – August 2025*

## Components / Skills

### Technical Skills / Tools

1. **Keras Machine Learning**
   - Keras is a high-level API designed for building and training neural networks. It provides a user-friendly interface that simplifies the process of creating machine learning models, especially for developers who prioritize ease of use and rapid prototyping.  
   - In this project, it was used to build a CNN capable of classifying brain MRI images into glioma, meningioma, or no tumor.  
   - The trained model is saved as a `.h5` file, which can be loaded directly to make predictions without retraining.  

2. **Python**
   - Core programming language for the project, used to implement the model, preprocessing, and GUI.  
   - Libraries included: `numpy` for numerical operations, `Pillow` for image processing, `tkinter` for GUI development.

3. **Dataset Handling**
   - The project uses the **Kaggle Brain Tumor MRI Dataset**, which includes over **5,000 scans** across training and testing sets.  
   - Images were loaded, resized, and normalized to prepare them for the CNN.  
   - The network has learned from this dataset to distinguish between glioma, meningioma, and no tumor scans.

4. **Tkinter GUI**
   - Built a simple, interactive interface for users to upload MRI images and view predictions.  
   - Demonstrates the integration of machine learning with user-facing software.

5. **Model Persistence**
   - Saving and reusing the trained model demonstrates understanding of practical deployment of machine learning in real-world applications.

---

### Soft Skills / Project Skills

1. **Problem Solving**
   - Tackled the challenge of classifying medical images and designing a workflow from dataset loading to prediction.  

2. **Attention to Detail**
   - Ensured images were correctly preprocessed (resized, normalized) for optimal model performance.  

3. **Project Management**
   - Organized scripts (`main.py`, `model.py`, `keras_ml_loader.py`, GUI) and maintained a clear separation between training and prediction.  

4. **Learning & Adaptability**
   - Researched and switched between TensorFlow and Keras when necessary, choosing the most efficient approach for this project.  

5. **Communication**
   - Designed the GUI and README to clearly convey functionality, usage instructions, and the motivation behind the project.

6. **Passion for Healthcare Technology**
   - Demonstrated commitment to using AI to improve brain scanning and early detection of brain tumors.


---

## About This Project

This project is a **Brain Tumor MRI Classifier** that I built to explore how machine learning can assist in medical imaging. It allows a user to upload a brain MRI scan and predicts whether it shows a glioma, meningioma, or no tumor.  

I designed this tool to be simple and interactive, using **Keras for the neural network** and **Tkinter for the GUI**, so anyone can test it on their own images without dealing with code.

---

## Dataset

The dataset I used comes from Kaggle: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)  

You can download it and place it in your project folder under `dataset/Brain Tumor MRI Dataset` to use it with the program.

The dataset is already interpereted by the machine learning model keras in a file called 'brain_tumor_model.h5'

---

## Why I Built This

Brain scanning technology has always been something that matters deeply to me. I’ve seen firsthand how critical early and accurate detection can be for people’s lives. Knowing that machine learning could help doctors interpret scans faster and more accurately inspired me to take on this project.  

This is more than just a coding exercise—it’s a personal step toward understanding **how AI can meaningfully assist in healthcare**, particularly in areas where speed and precision can make a real difference.

---

## How It Works

1. **Load the Trained Model** – The program uses a pre-trained CNN (`brain_tumor_model.h5`) to classify MRI images.  
2. **Select an Image** – Using the GUI, you can choose any brain MRI scan.  
3. **Prediction** – The program processes the image, normalizes it, and predicts the class with confidence.  
4. **Output** – The prediction is displayed alongside the uploaded image, making it easy to interpret.  

The model was trained on a dataset of brain MRI images, resized to 128×128 pixels, and normalized for best performance.

---

## How to Use

1. Make sure Python is installed on your computer.  
2. Install the required packages:

```bash
pip install keras pillow numpy
