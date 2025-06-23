# Depression Detection from Text using BERT + RNN

This project leverages **BERT embeddings** and **LSTM RNN** to detect signs of depression from social media text data. The deep learning model is trained to classify textual inputs into depressive or non-depressive categories, with potential for extension into severity classification.

## 📁 Project Structure

- `Depression_detection_RNNandBERT.ipynb` — Notebook to preprocess data, extract BERT embeddings, train the LSTM RNN model.
- `evaluate_model.ipynb` — Script to evaluate the model using classification metrics (accuracy, precision, recall, F1, confusion matrix).
- `inference.ipynb` — Interface to input custom text and run inference using the trained model.

## 🔍 Motivation

Mental health is a global concern. Early detection of depression through passive data (e.g., social media) could help in timely intervention. This project aims to classify text into depressive or non-depressive using powerful transformer and RNN-based deep learning techniques.

---

## 💡 Features

- **BERT for context-aware word embeddings**
- **LSTM RNN** to capture sequential patterns in user input
- **Binary classification** (Depressed vs Non-Depressed)
- **Explainability and visualization** planned for further development
- **Inference pipeline** for real-time usage

---

## 🧠 Model Architecture

- Pretrained **BERT** model: Extracts embeddings from text
- **LSTM RNN**: Processes BERT embeddings to understand temporal relationships
- Fully connected layers: Classifies into target categories

---

## 🧪 Requirements

Install dependencies via pip:

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn
```
---

## 📊 Dataset

> Note: Due to data privacy, the dataset used in this project is assumed to be preprocessed and available in the working directory.

The dataset consists of labeled social media text posts with depression labels (`0` for non-depressed, `1` for depressed).

---

## 🚀 How to Run

1. **Training the Model**

   Open and run `Depression_detection_RNNandBERT.ipynb`:

   * Loads dataset
   * Extracts BERT embeddings
   * Trains LSTM RNN
   * Saves the model

2. **Evaluating the Model**

   Run `evaluate_model.ipynb` to:

   * Load trained model
   * Visualize confusion matrix
   * Display precision, recall, F1-score, etc.

3. **Inference on New Text**

   Run `inference.ipynb`:

   * Enter custom input text
   * View prediction (Depressed/Not Depressed)

---

## 📈 Sample Output

```text
Input: "I feel empty and tired all the time."
Prediction: Depressed
```

---

## 🔧 Future Work

* Severity classification using PHQ-9 score ranges
* Explainable AI with attention visualizations
* Time-series tracking of user mood shifts
* Deployment as a web/mobile application

---

## 👨‍💻 Authors

* **Priyangshu Das**,**Anish Banerjee**,**Tanvir Hossain**,**Soumya Ghosh**
* B.Tech CSE Student, Meghnad Saha Institute of Technology

---

## 📜 License

This project is for academic and research purposes. Please cite appropriately when using or referencing.

