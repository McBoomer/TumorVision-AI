# This python tkiner GUI application allows users to select a brain MRI image,
# displays the image, and predicts whether it shows a brain tumor using a pre-trained CNN model
# located in 'brain_tumor_model.h5'. previously trained on Brain Tumor MRI dataset. using the main and model python files
# last edited on 08/20/2025 by Muadh Khan, Markham, Ontario
# requires: keras, tensorflow, pillow, numpy, tkinter
# Muadh Khan

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Config
IMG_SIZE = (128, 128)  # same as model input
MODEL_PATH = "brain_tumor_model.h5"
CLASS_NAMES = ["glioma", "meningioma", "notumor"]  # adjust to your dataset classes

# Load trained model
model = load_model(MODEL_PATH)

# Helper functions
def load_and_prepare_image(file_path):
    """Load image, resize, normalize, add batch dimension"""
    img = image.load_img(file_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def select_image():
    """Open file dialog, display image, and predict"""
    file_path = filedialog.askopenfilename(
        title="Select Brain MRI Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    # Display image in Tkinter
    pil_img = Image.open(file_path)
    pil_img = pil_img.resize((350, 350))
    tk_img = ImageTk.PhotoImage(pil_img)
    img_label.config(image=tk_img)
    img_label.image = tk_img
    img_label.pack(pady=20)
    remove_btn.pack(pady=10)
    select_btn.config(state=tk.DISABLED)

    # Predict class
    img_array = load_and_prepare_image(file_path)
    pred = model.predict(img_array)
    class_index = np.argmax(pred)
    confidence = pred[0][class_index] * 100

    # Set color based on confidence
    color = "green" if confidence >= 70 else "red"
    result_label.config(
        text=f"Prediction: {CLASS_NAMES[class_index]} ",
        fg="white"
    )
    percent_label.config(
        text=f"({confidence:.2f}%)",
        fg=color
    )
    percent_label.pack()

def remove_image():
    img_label.config(image="", text="")
    img_label.image = None
    result_label.config(text="", fg="white")
    percent_label.config(text="")
    select_btn.config(state=tk.NORMAL)
    remove_btn.pack_forget()

# Build Tkinter GUI
root = tk.Tk()
root.title("Brain MRI Classifier")
root.geometry("600x800")
root.configure(bg="black")

# Button to select image
select_btn = tk.Button(
    root,
    text="Select MRI Image",
    command=select_image,
    font=("Helvetica", 18, "bold"),
    bg="#444",
    fg="white",
    activebackground="#666",
    activeforeground="white",
    padx=20,
    pady=10
)
select_btn.pack(pady=20)

# Label to show the image
img_label = tk.Label(root, bg="black")
img_label.pack(pady=20)

# Label to show prediction
result_label = tk.Label(root, text="", font=("Helvetica", 22, "bold"), bg="black", fg="white")
result_label.pack(pady=10)

# Label for confidence percent
percent_label = tk.Label(root, text="", font=("Helvetica", 22, "bold"), bg="black")
percent_label.pack()

# Button to remove image
remove_btn = tk.Button(
    root,
    text="Remove Image",
    command=remove_image,
    font=("Helvetica", 16, "bold"),
    bg="#a00",
    fg="white",
    activebackground="#c33",
    activeforeground="white",
    padx=16,
    pady=8
)
remove_btn.pack_forget()

root.mainloop()