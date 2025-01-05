import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from preprocess import extract_enhanced_features, process_test_data, save_predictions, analyze_predictions
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils._tags")

def load_data():
    """Load processed data"""
    processed_dir = 'processed'
    
    # Load features and labels
    features = pd.read_csv(os.path.join(processed_dir, 'features.csv'), index_col=0)
    labels = pd.read_csv(os.path.join(processed_dir, 'labels.csv'), index_col=0)
    
    # Keep all features
    return features, labels['category']

def create_interaction_features(X):
    """Create interaction features for both training and test data"""
    X = X.copy()
    
    # Ratio features
    X['special_to_length'] = X['special_count'] / X['length']
    X['digit_to_length'] = X['digit_count'] / X['length']
    X['alpha_to_length'] = X['alpha_count'] / X['length']
    
    # Interaction features between category confidences
    confidence_cols = [col for col in X.columns if col.startswith('confidence_')]
    for i in range(len(confidence_cols)):
        for j in range(i+1, len(confidence_cols)):
            X[f'{confidence_cols[i]}_{confidence_cols[j]}'] = X[confidence_cols[i]] * X[confidence_cols[j]]
    
    return X

def train_and_evaluate_model(X, y):
    """Train and evaluate the model using Random Forest"""
    # Initialize label encoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Create features and scale them
    X = create_interaction_features(X)
    feature_names = X.columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(0))
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_encoded, 
        test_size=0.2, 
        random_state=42,
        stratify=y_encoded
    )
    
    # Create and train Random Forest
    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=10,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    
    # Train the model
    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)
    
    # Evaluate
    val_preds = model.predict(X_val)
    print(f"\nValidation Balanced Accuracy: {balanced_accuracy_score(y_val, val_preds):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_val, val_preds, target_names=le.classes_))
    
    return model, le, scaler, feature_names

def analyze_predictions(predictions, y_true):
    """Analyze predictions"""
    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true)
    y_pred_encoded = le.transform(predictions)
    
    print("\nClassification Report:")
    print(classification_report(y_true_encoded, y_pred_encoded, target_names=le.classes_))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true_encoded, y_pred_encoded)
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    """Main function"""
    print("Loading classification data...")
    
    # Load training data
    X, y = load_data()
    
    # Display class distribution
    print("\nClass distribution in training:")
    for category, count in y.value_counts().items():
        print(f"{category}: {count} ({count/len(y)*100:.1f}%)")
    
    # Train and evaluate model
    model, label_encoder, scaler, feature_names = train_and_evaluate_model(X, y)
    
    # Process test data
    print("\nProcessing test data...")
    test_file = os.path.join('data', 'test-classification-round2.dat')
    X_test, test_usernames = process_test_data(test_file)
    
    # Create same features for test data
    X_test = create_interaction_features(X_test)
    X_test = X_test.reindex(columns=feature_names, fill_value=0)
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test.fillna(0))
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_test_scaled)
    predictions = label_encoder.inverse_transform(predictions)
    
    # Display prediction distribution
    print("\nPrediction distribution:")
    pred_series = pd.Series(predictions)
    for category, count in pred_series.value_counts().items():
        print(f"{category}: {count} ({count/len(predictions)*100:.1f}%)")
    
    # Create predictions dictionary
    predictions_dict = dict(zip(test_usernames, predictions))
    
    # Save predictions
    output_file = 'prediction-classification-round2.json'
    with open(output_file, 'w') as f:
        json.dump(predictions_dict, f, indent=4)

if __name__ == '__main__':
    main() 