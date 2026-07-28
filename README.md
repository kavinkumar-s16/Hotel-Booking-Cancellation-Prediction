# Hotel Booking Cancellation Prediction

## Project Overview

Hotel booking cancellations cause revenue loss and difficulty in resource management. 

This project uses Machine Learning techniques to predict whether a hotel booking will be cancelled or not based on customer and booking details.

The model learns patterns from historical booking data and predicts cancellation probability for new bookings.

---

# Problem Statement

To build a Machine Learning model that can predict hotel booking cancellations using customer information, booking details, and reservation features.

---

# Dataset

Dataset used:

Hotel Booking Demand Dataset

The dataset contains booking information such as:

- Hotel type
- Lead time
- Arrival details
- Customer type
- Deposit type
- Previous cancellations
- Booking changes
- Special requests
- Room information

Target Variable:

```
is_canceled
```

Values:

```
0 - Booking Not Cancelled
1 - Booking Cancelled
```

---

# Technologies Used

## Programming Language

- Python

## Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

# Machine Learning Workflow

```
Data Collection

        ↓

Data Exploration

        ↓

Data Cleaning

        ↓

Exploratory Data Analysis

        ↓

Feature Encoding

        ↓

Train-Test Split

        ↓

Model Training

        ↓

Model Evaluation

        ↓

Deployment using Streamlit
```

---

# Exploratory Data Analysis

Performed:

- Missing value analysis
- Statistical analysis
- Distribution analysis
- Correlation analysis
- Feature visualization


Visualization techniques used:

- Histogram
- Count Plot
- Box Plot
- Heatmap

---

# Important Features

## Hotel Type

Identifies whether the booking belongs to City Hotel or Resort Hotel.

## Lead Time

Number of days between booking date and arrival date.

## ADR

Average Daily Rate of the booking.

## Customer Type

Defines customer categories such as transient, group, and contract customers.

## Previous Cancellations

Number of previous cancelled bookings by the customer.

## Deposit Type

Payment/deposit policy used during booking.

## Booking Changes

Number of modifications made after reservation.

## Special Requests

Additional customer requirements during booking.

---

# Machine Learning Algorithms Used

## 1. Logistic Regression

Used as a baseline classification model.

## 2. Decision Tree

Used to understand decision-based classification.

## 3. Random Forest

Final selected model because it provided better performance and reduced overfitting.

---

# Model Performance

Random Forest achieved approximately:

```
Accuracy: 85%
```

Evaluation metrics:

- Accuracy Score
- Confusion Matrix
- Classification Report

---

# Deployment

The trained model was deployed using Streamlit.

The application allows users to enter booking details and predicts:

```
Booking will be Cancelled

or

Booking will NOT be Cancelled
```

---

# Project Demo

Streamlit Interface:

(Add screenshot here)

---

# Installation

Clone repository:

```bash
git clone https://github.com/yourusername/Hotel-Booking-Cancellation-Prediction.git
```

Install requirements:

```bash
pip install -r requirements.txt
```

Run application:

```bash
streamlit run app.py
```

---

# Future Improvements

- Handle class imbalance using advanced techniques
- Add probability-based predictions
- Deploy using cloud platforms
- Improve model performance using advanced algorithms

---

# Author

Kavin Kumar

Computer Science Engineering Student