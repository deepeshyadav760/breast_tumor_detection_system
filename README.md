# Tumor Detection in MRI Images
This project focuses on detecting and classifying tumors in MRI images into three categories: 
 1. Non-Cancer
 2. Early Stage Cancer
 3. Middle Stage Cancer

The project uses a Convolutional Neural Network (CNN) for image classification and incorporates a Tkinter-based User Interface for ease of use.
Features
Deep Learning Model:
   1.Custom CNN architecture for MRI image classification.
   2.Trained to identify tumor stages with high accuracy.

Preprocessing Pipeline:
   1.Resizing, normalization, and augmentation of input MRI images.

Graphical User Interface (GUI):
   1.Built using Tkinter for user-friendly interaction.
   2.Allows users to upload MRI images, run predictions, and view results directly in the application.

Requirements
To run this project, ensure you have the following dependencies installed:
Python 3.7+
TensorFlow 2.x
NumPy
Pillow
Tkinter (comes pre-installed with Python)
Matplotlib (optional, if visualizations are included)

Install all dependencies using:
bash
Copy code
Usage
Running the Application

Clone the repository to your local system:
bash
Copy code
git clone https://github.com/yourusername/tumor-detection-mri.git

Navigate to the project directory:
bash
Copy code
cd tumor-detection-mri

Run the application:
bash
Copy code
python app.py
The Tkinter-based GUI will open.

Using the GUI
Upload Image:
Click on the "Upload Image" button to browse and select an MRI image.

Predict Tumor:
Click on the "Predict" button. The application will process the image and display the classification result.

View Results of the diagonis analysis:
The prediction will be displayed on the interface, showing whether the image corresponds to Non-Cancer, Early Stage Cancer, or Middle Stage Cancer.
