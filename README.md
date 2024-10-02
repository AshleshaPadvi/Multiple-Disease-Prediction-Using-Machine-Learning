# Multiple-Disease-Prediction-Using-Machine-Learning

This repository contains the implementation of a machine learning framework for predicting the prognosis of multiple diseases, including Diabetes Mellitus, Myocardial Infarction and Parkinson's Disease. The project leverages Support Vector Machines (SVM) and Logistic Regression algorithms to ensure high accuracy and model interpretability, focusing on disease prediction using medical datasets.

Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
  
Introduction

This project aims to build an intelligent machine learning model that can predict the prognosis of multiple diseases based on patient medical records and history. The diseases covered include:

1. Diabetes Mellitus
2. Myocardial Infarction
3. Parkinson's Disease

The model is built using machine learning algorithms to provide early predictions and help in the treatment planning process for medical professionals.

Features

- Predicts the likelihood of multiple diseases based on input features.
- Uses SVM and Logistic Regression for disease prediction.
- Designed with an emphasis on **model interpretability.
- Provides insights into medical features contributing to disease prognosis.
- Easy-to-understand and highly scalable for other diseases.

Technologies Used

- Programming Language: Python
- Libraries: 
  - Machine Learning: `scikit-learn`
  - Data Processing: `pandas`, `NumPy`
  - Visualization: `matplotlib`, `seaborn`
- Front-end:
  - `NextJs`, `ReactJs`, `Tailwind`, `Shadcn`
- Authentication**: `Clerk`

Project Architecture

```bash
|-- dataset/                 # Folder for storing datasets
|-- src/                     # Source code folder
|   |-- models/              # Machine learning models (SVM, Logistic Regression)
|   |-- preprocessing/       # Data preprocessing scripts
|   |-- utils/               # Utility functions for data handling
|-- README.md                # Project overview
|-- requirements.txt         # Required dependencies
```

Key Components

- Models: Implementation of machine learning models like SVM and Logistic Regression.
- Data Preprocessing: Handles missing data, normalization, and feature selection.
- User Interface: A user-friendly interface for inputting patient data and obtaining predictions.

Dataset

- The project uses publicly available medical datasets for training the machine learning models.
- Each dataset contains relevant features related to the diseases being predicted, such as blood glucose levels, age, ECG signals, etc.
- Note: The datasets should be placed in the `dataset/` folder before running the project.

Model Performance

- Support Vector Machine and Logistic Regression were chosen based on their ability to handle complex patterns in medical data.
- Accuracy:
  - Diabetes Mellitus: 85%
  - Myocardial Infarction: 80%
  - Parkinson's Disease: 88%
- Evaluation Metrics: The models were evaluated using **Accuracy**, **Precision**, **Recall**, and **F1-score**.

Installation

To run this project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Multiple-Disease-Prediction-Using-Machine-Learning.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Multiple-Disease-Prediction-Using-Machine-Learning
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Make sure you have the necessary datasets in the `dataset/` folder.

Usage

To use the project, follow these steps:

1. Run the preprocessing script to clean and normalize the data.
   
   ```bash
   python src/preprocessing/clean_data.py
   ```

2. Train the machine learning model using the following command:

   ```bash
   python src/models/train_model.py
   ```

3. Once the model is trained, you can start the web interface by running:

   ```bash
   npm run dev  # For the NextJs-based front-end
   ```

4. Input the medical data via the UI, and the model will predict the likelihood of diseases.

Results

- After training, the machine learning models will output the probability of a patient having a certain disease based on their medical history.
- You can visualize the model's performance using **confusion matrices**, **ROC curves**, and **other metrics**.

Contributing

If you wish to contribute to this project, feel free to fork the repository and submit a pull request with your changes.

1. Fork the repository
2. Create a new feature branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a pull request

License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
