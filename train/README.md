# Image Classifier with TensorFlow

This project demonstrates how to train a Convolutional Neural Network (CNN) model for classifying images using TensorFlow. The model is trained on a custom dataset with data augmentation applied to improve generalization. The code also includes functions for training, evaluating, and saving the model.

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

## Project Structure


## Dataset

The dataset used for training is **not included** in this repository. To train the model, you need to prepare your own dataset. The dataset should be organized in the following structure:


Each class should have its own folder containing the relevant images. You can use your own dataset or find a similar one for training. Make sure the images are properly organized into class subdirectories.

### Example Directory Structure:


## Dependencies

The required dependencies for this project are:

- **TensorFlow**: For building and training the CNN model.
- **Numpy**: For numerical operations.
- **Matplotlib**: For plotting training history.
- **Seaborn**: For improved visualization of training results.
- **Scikit-learn**: For generating classification reports and confusion matrices.

Install the dependencies by running:

```bash
pip install -r requirements.txt
