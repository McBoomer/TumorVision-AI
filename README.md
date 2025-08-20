# Brain Tumor MRI Classifier

### By Muadh Khan  
*Markham, Ontario – August 2025*

---

## About This Project

This project is a **Brain Tumor MRI Classifier** that I built to explore how machine learning can assist in medical imaging. It allows a user to upload a brain MRI scan and predicts whether it shows a glioma, meningioma, or no tumor.  

I designed this tool to be simple and interactive, using **Keras for the neural network** and **Tkinter for the GUI**, so anyone can test it on their own images without dealing with code.

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
