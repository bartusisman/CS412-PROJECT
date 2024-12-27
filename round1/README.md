# Instagram Account Classification and Like Prediction

This project consists of two main components:

1. Account Classification: Predicts the category of Instagram accounts (e.g., Tech, Food, Sports, etc.)
2. Like Count Prediction: Predicts the number of likes a post will receive

## Project Structure

```
round1/
├── data/
│   ├── training-dataset.jsonl.gz     # Training dataset with user profiles and posts
│   ├── train-classification.csv      # Training labels for account classification
│   ├── test-classification-round1.dat # Accounts to be classified
│   └── test-regression-round1.jsonl   # Posts for like count prediction
├── preprocess.py                     # Data preprocessing for classification
├── train_and_predict.py             # Account category prediction
├── regression.py                     # Like count prediction
└── README.md                        # This file
```

## Requirements

- Python 3.8+
- Required packages:
  ```
  pandas>=1.3.0
  numpy>=1.19.0
  scikit-learn>=0.24.0
  imbalanced-learn>=0.8.0
  ```

## Installation

1. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Account Classification

Run the following commands in sequence:

```bash
python preprocess.py
python train_and_predict.py
```

This will:

- Process the training data and extract features
- Train a balanced random forest classifier
- Generate predictions for test accounts
- Save results in `prediction-classification-round1.json`

### 2. Like Count Prediction

Run:

```bash
python regression.py
```

This will:

- Load and process post data
- Train a random forest regressor
- Generate like count predictions
- Save results in `prediction-regression-round1.json`

## Output Files

1. `prediction-classification-round1.json`: Contains account category predictions

   ```json
   {
       "username1": "category1",
       "username2": "category2",
       ...
   }
   ```

2. `prediction-regression-round1.json`: Contains like count predictions
   ```json
   {
       "post_id1": predicted_likes1,
       "post_id2": predicted_likes2,
       ...
   }
   ```

## Features

### Classification Features

- Username characteristics (length, numbers, special characters)
- Keyword-based features for each category
- Text patterns and indicators

### Regression Features

- Media type
- Comment count
- Caption length
- Hashtag count
- Mention count

## Models

1. Classification: Balanced Random Forest Classifier

   - Handles class imbalance
   - Provides feature importance
   - Uses cross-validation for evaluation

2. Regression: Random Forest Regressor
   - Handles non-linear relationships
   - Robust to outliers
   - Provides feature importance
