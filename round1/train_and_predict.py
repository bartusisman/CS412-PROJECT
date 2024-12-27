import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.ensemble import BalancedRandomForestClassifier
from preprocess import extract_username_features
import json
import os
import re

def clean_username(username):
    """Clean username by removing extra text and newlines"""
    return re.sub(r'&entry\.[0-9]+=.*\n?', '', username).lower().strip()

def load_data():
    """Load processed features and labels"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(current_dir, 'processed')
    
    features = pd.read_csv(os.path.join(processed_dir, 'features.csv'), index_col=0)
    labels = pd.read_csv(os.path.join(processed_dir, 'labels.csv'), index_col=0)
    
    with open(os.path.join(processed_dir, 'category_mapping.json'), 'r') as f:
        category_mapping = json.load(f)
    
    return features, labels.iloc[:, 0], category_mapping

def load_test_usernames():
    """Load usernames to predict"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(current_dir, 'data', 'test-classification-round1.dat')
    
    with open(test_path, 'r') as f:
        usernames = [line.strip() for line in f]
    
    return usernames

def main():
    print("Loading data...")
    X, y, category_mapping = load_data()
    
    # Convert categories to numeric
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Use balanced random forest
    print("\nTraining balanced random forest...")
    model = BalancedRandomForestClassifier(
        n_estimators=1000,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        sampling_strategy='auto'
    )
    
    # Evaluate with stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y_encoded, cv=cv, scoring='balanced_accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Train final model
    print("\nTraining final model...")
    model.fit(X, y_encoded)
    
    # Print feature importance
    print("\nTop 10 Most Important Features:")
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance_df.head(10).iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    # Load and process test data
    test_usernames = load_test_usernames()
    test_usernames = [clean_username(u) for u in test_usernames]
    
    # Extract features for test data
    test_features = pd.DataFrame([extract_username_features(u) for u in test_usernames])
    test_features = test_features[X.columns]  # Ensure same column order
    
    # Make predictions
    print("\nMaking predictions...")
    pred_encoded = model.predict(test_features)
    pred_categories = le.inverse_transform(pred_encoded)
    
    # Create predictions dictionary
    predictions = {}
    for username, category in zip(test_usernames, pred_categories):
        predictions[username] = category
    
    # Save predictions
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'prediction-classification-round1.json'
    )
    
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=4)
    
    # Analyze predictions
    pred_dist = pd.Series(pred_categories).value_counts()
    print("\nPrediction distribution:")
    for category, count in pred_dist.items():
        print(f"{category}: {count} ({count/len(predictions)*100:.1f}%)")
    
    print("\nSample predictions:")
    for username in list(predictions.keys())[:5]:
        print(f"{username}: {predictions[username]}")

if __name__ == "__main__":
    main() 