# AgriVision: Intelligent Plant Disease Diagnostics

AgriVision is an advanced deep-learning project designed to provide instant, automated identification of plant diseases from leaf images. By leveraging high-performance computer vision, this system empowers farmers and researchers to diagnose crop issues accurately, facilitating timely intervention and reducing crop loss.

## 📊 Dataset & Configuration

This project utilizes the **PlantVillage Dataset**, a comprehensive repository containing over 54,000 images of healthy and diseased crop leaves. The dataset is accessed via the Kaggle API.

### **Kaggle Setup Instructions**

To access the dataset during the model training phase, a `kaggle.json` file is required for authentication.

1. Go to your **Kaggle Account** settings.
2. Scroll to the **API** section and click **"Create New API Token"**.
3. Download the `kaggle.json` file.
4. Upload this file into your environment when prompted to allow the script to download the **abdallahalidev/plantvillage-dataset** automatically.

## ⚙️ Technical Workflow & Architecture

### **Core Architecture: MobileNetV2**

The system is built on the **MobileNetV2** architecture, a powerful convolutional neural network (CNN) optimized for mobile and edge devices.

* **Why MobileNetV2?** Unlike heavier models like ResNet or VGG, MobileNetV2 uses depthwise separable convolutions to significantly reduce the number of parameters and computational cost without sacrificing high accuracy.
* **Transfer Learning:** The model utilizes weights pre-trained on the **ImageNet** dataset. This allows the AI to start with a sophisticated understanding of shapes and textures, which is then fine-tuned specifically for botanical patterns.
* **Fine-Tuning Strategy:** To optimize performance, the first 100 layers of the base model are frozen to preserve general feature extraction, while the remaining layers are unfrozen to adapt specifically to plant disease features.

### **Data Augmentation**

To improve the model's ability to generalize to real-world photos (which may have varied lighting or angles), **ImageDataGenerator** was used to implement data augmentation.

* **Techniques used:** The training data was subjected to 20-degree rotations, 20% width/height shifts, and horizontal flips.
* **Importance:** This process artificially expands the dataset, preventing the model from "memorizing" specific training images (overfitting) and ensuring it recognizes leaves regardless of their orientation in a user's photo.

## 📈 Performance Metrics

The dataset was split into **80% Training**, **10% Validation**, and **10% Testing** sets using a fixed seed for reproducibility.

| Metric | Training Set | Validation Set | Testing Set |
| --- | --- | --- | --- |
| **Accuracy** | 98.21% | 95.49% | 94.50% |
| **Loss** | 0.0232 | 0.0465 | 0.0451 |
| **Precision** | 0.9726 | 0.9567 | 0.9257 |
| **Recall** | 0.9615 | 0.9439 | 0.9144 |
| **F1-Score** | 0.9670 | 0.9503 | 0.9200 |

*(Note: F1-Scores calculated using the harmonic mean of Precision and Recall as per standard evaluation practices.)*

## ⚠️ Scope & Limitations

While highly accurate, the system operates within a specific diagnostic scope:

* **Species & Classes:** The model can detect **38 distinct health classes** across **14 different plant species**.
* **Detected Plants:** Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper (Bell), Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.
* **Diagnostic Limitation:** The AI is specialized for these 14 species; it may provide unreliable results for plants outside this scope.
* **Confidence Guard:** The system includes a reliability check; predictions with a confidence score lower than **60%** are flagged as potentially unreliable to ensure user safety.

## 🔗 Live Deployment

Access the live diagnostic tool here: **[[Streamlit Link Here](https://agrivision-ai.streamlit.app/)]**
