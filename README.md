# Smart Crop Disease Prediction

Smart Crop Disease Prediction : an advanced deep-learning project designed to provide instant, automated identification of plant diseases from leaf images. By leveraging high-performance computer vision and generative AI–powered recommendations, this system empowers farmers, agronomists, and researchers to diagnose crop issues accurately, take timely action, and significantly reduce crop loss.

---

## 📊 Dataset & Configuration

This project utilizes the **PlantVillage Dataset**, a large-scale, open-source repository containing over **54,000 labeled images** of healthy and diseased crop leaves. The dataset is accessed programmatically using the **Kaggle API**, enabling automated and reproducible model training.

### **Kaggle Setup Instructions**

To download the dataset during training, a `kaggle.json` file is required for authentication.

1. Log in to your **Kaggle account**.
2. Navigate to **Account Settings**.
3. Scroll to the **API** section and click **"Create New API Token"**.
4. Download the generated `kaggle.json` file.
5. Upload this file into your working environment when prompted. The script will then automatically download the dataset: **abdallahalidev/plantvillage-dataset**.

---

## ⚙️ Technical Workflow & Architecture

### **Core Architecture: MobileNetV2**

AgriVision is built on **MobileNetV2**, a lightweight yet highly effective convolutional neural network (CNN) optimized for real-time and resource-constrained environments.

**Why MobileNetV2?**

* Uses **depthwise separable convolutions**, drastically reducing computational cost.
* Requires fewer parameters compared to heavier models such as VGG or ResNet.
* Delivers high accuracy while remaining suitable for web and mobile deployment.

### **Transfer Learning Strategy**

* Pre-trained weights from the **ImageNet** dataset are used to initialize the model.
* The **first 100 layers are frozen** to retain generic feature extraction (edges, textures, shapes).
* Remaining layers are **fine-tuned** to specialize in plant disease characteristics such as spots, discoloration, and leaf deformities.

---

## 🔄 Data Augmentation

To enhance robustness and generalization to real-world conditions, **ImageDataGenerator** was employed for data augmentation.

**Applied Techniques:**

* Rotation up to **20 degrees**
* Width and height shifts of **20%**
* **Horizontal flipping**

**Impact:**
This approach increases dataset diversity, minimizes overfitting, and ensures reliable predictions across varying lighting conditions, angles, and backgrounds.

---

## 📈 Model Performance Metrics

The dataset was split using a fixed random seed for reproducibility:

* **80% Training**
* **10% Validation**
* **10% Testing**

### **Evaluation Results (Final Model)**

| Dataset        | Accuracy | Precision | Recall | F1-Score |
| -------------- | -------- | --------- | ------ | -------- |
| **Training**   | 99.68%   | 0.9970    | 0.9968 | 0.9969   |
| **Validation** | 99.04%   | 0.9915    | 0.9902 | 0.9909   |
| **Testing**    | 99.05%   | 0.9912    | 0.9897 | 0.9905   |

These results demonstrate strong generalization performance, minimal overfitting, and high diagnostic reliability across unseen data.

---

## 🤖 AI-Powered Disease Suggestions (Groq API Integration)

Beyond disease classification, AgriVision integrates the **Groq API** to provide **intelligent, context-aware recommendations** when a disease is detected.

### **How It Works**

* Once the CNN predicts a disease class, the result is passed to the **Groq LLM API**.
* The model generates **actionable suggestions**.

### **Benefits**

* Transforms raw predictions into **farmer-friendly insights**.
* Reduces dependency on agricultural experts for basic diagnostics.
* Enhances decision-making by coupling vision intelligence with generative AI.

If the detected disease confidence is below **60%**, the system flags the output as **unreliable** and avoids generating misleading recommendations.

---

## ⚠️ Scope & Limitations

* **Supported Classes:** 38 disease and healthy classes
* **Supported Species (14):**
  Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

**Limitations:**

* Predictions for plant species outside this list may be inaccurate.
* Performance may degrade on images with extreme blur, occlusion, or non-leaf objects.

---

## 🔗 Live Deployment

The application is deployed using **Streamlit**, enabling real-time disease detection through a simple web interface.

👉 **Live App:** [https://agrivision-ai.streamlit.app/](https://agrivision-ai.streamlit.app/)

---

## ✅ Conclusion

AgriVision combines **deep learning**, **transfer learning**, and **generative AI** to deliver an end-to-end intelligent plant disease diagnostic system. With high accuracy, real-time inference, and actionable AI-driven recommendations, the platform serves as a practical and scalable solution for modern agriculture.
