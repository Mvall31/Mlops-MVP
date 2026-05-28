# Hugging Face Inception

This is an index of what you will learn following the tutorial `STEP-by-STEP_GUIDE.md` supported by the code in this repository.

## 1. Introduction
1. **What is Hugging Face?**  
   - Brief overview of the Hugging Face ecosystem  
   - Key features relevant to MLOps
2. **Why Use Hugging Face for MLOps?**  
   - Benefits (collaboration, versioning, community tools)  
   - Comparison with traditional MLOps workflows

---

## 2. Getting Started
1. **Prerequisites**  
   - Required libraries: `datasets`, `scikit-learn`, `huggingface_hub`, `gradio` (or `streamlit`)  
   - Python version, environment setup
2. **Creating a Hugging Face Account**  
   - Steps to sign up  
   - Generating an access token
3. **Configuring Local Environment**  
   - Installing packages (`pip install ...`)  
   - Authenticating with Hugging Face Hub (using access tokens)

---

## 3. Working with Datasets on Hugging Face
1. **Overview of Hugging Face Datasets Hub**  
   - Searching for a dataset on the Hub  
   - Available features (like dataset versioning, metadata)
2. **Loading the Iris Dataset**  
   - Option A: Load from `datasets` library (if published)  
   - Option B: Load from Scikit-Learn’s built-in datasets and then prepare to push it to HF
3. **Inspecting the Dataset**  
   - Basic data exploration (feature names, target classes, data shape)

---

## 4. Data Preprocessing
1. **Splitting Data**  
   - Train/test split  
   - Potential train/validation/test approach
2. **Feature Engineering**  
   - Scaling or normalization, if needed  
   - Encoding steps if there were categorical variables
3. **Preparing for Model Training**  
   - Setting up `X` (features) and `y` (labels)

---

## 5. Model Training and Evaluation
1. **Creating a Logistic Regression Model in Scikit-Learn**  
   - Setting up hyperparameters (e.g., `C`, `max_iter`)  
   - Fitting the model on the training set
2. **Evaluating the Model**  
   - Accuracy, precision, recall, or any relevant metrics  
   - Printing confusion matrix, classification report

---

## 6. Uploading the Dataset to Hugging Face
1. **Why Upload Your Dataset?**  
   - Collaborative purposes, reproducibility  
   - Share the exact training data or a subset
2. **Using `huggingface_hub` to Create a New Dataset Repository**  
   - Repository structure for a dataset  
   - Steps to push the data (e.g., `repo.git_add()`, `repo.git_commit()`, `repo.git_push()`)

---

## 7. Saving and Uploading the Model to Hugging Face
1. **Serializing Your Model**  
   - Using Pickle or Joblib to save the trained model locally
2. **Creating a Model Repository**  
   - Structure for code, model artifacts, README  
   - Using `huggingface_hub` Python functions to initialize and push
3. **Versioning and Model Card**  
   - Adding metadata to the Model Card (e.g., usage, example code snippets)  
   - Good practices for describing your model’s purpose, performance, and limitations

---

## 8. Creating a Hugging Face Space for Deployment
1. **Overview of Spaces**  
   - Types of Spaces (Gradio, Streamlit, static web apps, etc.)  
   - Pros and cons for each approach
2. **Building a Demo with Gradio (or Streamlit)**  
   - Minimal code example to load the model from the Hub  
   - Creating interactive UI for inference (e.g., predict Iris species from flower measurements)
3. **Deploying Your Space**  
   - Linking the code to a Space on Hugging Face  
   - Managing Space settings, hardware requirements
4. **Testing the Space**  
   - Verifying predictions and performance  
   - Sharing the demo link with others
