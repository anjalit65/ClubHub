
# **Google Lens Alternative - Image Similarity Search**

## **Overview**

This repository provides a framework for developing an image similarity search engine, designed as an alternative to Google Lens. Various methods, including Convolutional Neural Networks (CNNs), Deep Metric Learning, Vision Transformers (CLIP), Locality Sensitive Hashing (LSH), and Autoencoders, are used to enable robust and efficient image retrieval.

---

## **Table of Contents**

- [Project Objective](#project-objective)
- [Methods Implemented](#methods-implemented)
  - [Pre-trained CNN (ResNet) + Nearest Neighbor Search](#pre-trained-cnn-resnet--nearest-neighbor-search)
  - [Deep Metric Learning (Siamese Network)](#deep-metric-learning-siamese-network)
  - [Vision Transformer (CLIP)](#vision-transformer-clip)
  - [Hashing (Locality Sensitive Hashing)](#hashing-locality-sensitive-hashing)
  - [Autoencoder](#autoencoder)
- [Usage](#usage)
- [Performance Comparison](#performance-comparison)
- [Insights and Recommendations](#insights-and-recommendations)
- [Contributors](#contributors)

---

## **Project Objective**

The goal of this project is to implement an image similarity search solution, exploring various approaches to determine the most effective method for different use cases. Each method has been evaluated based on accuracy, speed, and retrieval performance.

---

## **Methods Implemented**

### **1. Pre-trained CNN (ResNet) + Nearest Neighbor Search**
- **Description**: Utilizes a pre-trained ResNet50 model for feature extraction and FAISS for fast similarity search.
- **Best Use Case**: High-precision, detailed image retrieval.
- **Strengths**: High accuracy and efficient indexing with FAISS.

### **2. Deep Metric Learning (Siamese Network)**
- **Description**: Uses a Siamese Network for metric learning to capture subtle variations in image similarities.
- **Best Use Case**: Fine-grained similarity in variations.
- **Strengths**: Good precision, especially for similar image types.

### **3. Vision Transformer (CLIP)**
- **Description**: Leverages CLIP’s ViT-based embeddings for both semantic and visual similarity.
- **Best Use Case**: Content-rich or mixed datasets.
- **Strengths**: Effective for complex images with mixed content (e.g., text and objects).

### **4. Hashing (Locality Sensitive Hashing)**
- **Description**: Combines ResNet features with FeatureHasher for quick similarity retrieval.
- **Best Use Case**: Real-time search on simple images.
- **Strengths**: Fast processing, though less accurate.

### **5. Autoencoder**
- **Description**: Employs an autoencoder for dimensionality reduction and latent space similarity.
- **Best Use Case**: General similarity on simple datasets.
- **Strengths**: Good generalization for simple images, though lower accuracy.


---

## **Usage**

1. Place images in a folder named `test_images` within the root directory.
2. Run the main script to perform similarity searches across all methods:
   ```bash
   python lens.py
   ```
3. Results will display the top 5 matches for each method based on a specified query image.

---

## **Performance Comparison**

Each method was evaluated for:
- **Accuracy**: Precision in finding similar images.
- **Speed**: Time taken to retrieve similar images.
- **Best Use Case**: Scenarios suited for each method.

| Method              | Best Use Case                                | Speed     | Accuracy | Summary                                |
|---------------------|----------------------------------------------|-----------|----------|----------------------------------------|
| ResNet + k-NN       | High-precision, detailed image retrieval     | Moderate  | High     | Best for most applications needing accuracy |
| Siamese Network     | Fine-grained similarity                      | High      | High     | Good for nuanced image distinctions   |
| CLIP                | Semantic similarity                          | Moderate  | Moderate-High | Useful for complex datasets           |
| LSH                 | Real-time, low-complexity image search       | Very High | Low      | Fastest, best for basic searches      |
| Autoencoder         | General similarity on simple datasets        | High      | Low      | Limited to simple/generalized searches |

---

## **Insights and Recommendations**

- **ResNet with FAISS** or **Siamese Networks** provide the best trade-off between accuracy and efficiency for most use cases.
- **CLIP** works well for images with diverse content types (e.g., objects and text).
- **LSH** and **Autoencoder** are best suited for simple, real-time tasks where high precision is less critical.

---

## **Contributors**

- [Anjali Tripathi](https://github.com/anjalit65)

---

Feel free to reach out for questions or suggestions!

---
