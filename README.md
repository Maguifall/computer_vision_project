Models used: CNN1 (PyTorch), CNN2 (TensorFlow)

1. Introduction

Brain tumors are life-threatening conditions that require timely and accurate diagnosis. Mag-
netic Resonance Imaging (MRI) plays a crucial role in identifying and classifying tumor types.
In this project, we developed a web-based deep learning system for automatic brain tumor clas-
sification. The system uses convolutional neural networks (CNNs) to categorize MRI images
into four classes: glioma, meningioma, pituitary tumor, and no tumor.

2. Dataset and Preprocessing

The dataset consists of pre-labeled MRI scans representing the four target classes. All im-
ages were resized to a uniform resolution, normalized for consistency, and split into training,
validation, and testing sets. This ensured the models were trained and evaluated effectively.

3. Models and Training

Two CNN architectures were implemented and trained for this task:
• CNN1 (PyTorch): A moderately deep network with four convolutional layers, followed
by batch normalization, max pooling, dropout, and two dense layers.
• CNN2 (TensorFlow): A deeper architecture with multiple convolutional blocks, L2
regularization, dropout layers, and dense layers to increase learning capacity and general-
ization.

Both models were trained using the Adam optimizer and categorical cross-entropy loss.
While CNN1 showed strong performance, CNN2 also delivered promising results and represents
a solid base for further improvements.

4. Web Application

A web-based interface was developed to facilitate user interaction with the models. The appli-
cation allows users to upload an MRI image and select a classification model. It then displays
the predicted tumor type along with class probabilities.

Example output:

Class: pituitary
Confidence: 92.45%
All Probabilities:
- glioma: 3.12%
- meningioma: 2.05%
- notumor: 2.38%
- pituitary: 92.45%
- 
5. Conclusion and Future Work

This project demonstrates the viability of CNN-based systems for brain tumor image classifica-
tion. Both models serve as solid baselines, and their integration into a web application improves
accessibility for users.
While CNN1 achieved strong performance, CNN2 presents a promising foundation that could
be further enhanced. Future work may focus on applying data augmentation techniques—such
as rotation, flipping, and intensity variation—to increase the diversity of the training set and
help the model generalize better. Moreover, leveraging transfer learning with pre-trained convo-
lutional networks could significantly improve CNN2’s performance by transferring useful features
learned from large-scale image datasets, particularly when labeled medical data is limited.
