import os
import json
import numpy as np
import pandas as pd
from preprocess import extract_enhanced_features, save_predictions

def predict_category(features):
    """Predict category based on features"""
    categories = ['Tech', 'Food', 'Sports', 'Gaming', 'Fashion', 'Entertainment', 'Travel', 'Art', 'Health and Lifestyle']
    
    # Find the category with highest confidence
    max_confidence = -1
    predicted_category = None
    
    for category in categories:
        category_lower = category.lower().replace(' and ', '_').replace(' ', '_')
        confidence_key = f'confidence_{category_lower}'
        if confidence_key in features and features[confidence_key] > max_confidence:
            max_confidence = features[confidence_key]
            predicted_category = category
    
    # If no clear category found, use pattern-based features
    if max_confidence == 0 or predicted_category is None:
        if features.get('has_tech', 0) == 1:
            predicted_category = 'Tech'
        elif features.get('has_gaming', 0) == 1:
            predicted_category = 'Gaming'
        elif features.get('has_food', 0) == 1:
            predicted_category = 'Food'
        elif features.get('has_sports', 0) == 1:
            predicted_category = 'Sports'
        elif features.get('has_fashion', 0) == 1:
            predicted_category = 'Fashion'
        elif features.get('has_art', 0) == 1:
            predicted_category = 'Art'
        elif features.get('has_travel', 0) == 1:
            predicted_category = 'Travel'
        elif features.get('has_health', 0) == 1:
            predicted_category = 'Health and Lifestyle'
        elif features.get('has_entertainment', 0) == 1:
            predicted_category = 'Entertainment'
        else:
            predicted_category = 'Tech'  # Default category
    
    return predicted_category

def predict_engagement(features):
    """Predict engagement score based on features"""
    base_score = 1000
    
    # Adjust based on username characteristics
    if features.get('length', 0) > 10:
        base_score += 500
    if features.get('word_count', 0) > 1:
        base_score += 300
    if features.get('has_special', 0) == 1:
        base_score -= 200
    if features.get('is_turkish', 0) == 1:
        base_score += 400
    
    # Add some randomness
    engagement_score = int(base_score + np.random.normal(0, 200))
    return max(100, min(10000, engagement_score))  # Clip to reasonable range

def main():
    """Main function"""
    print("Processing test data...")
    test_file = os.path.join('round3', 'data', 'Test Classification Round 3.dat')
    
    # Load and process test data
    with open(test_file, 'r') as f:
        test_usernames = [line.strip() for line in f]
    
    # Make predictions
    cls_predictions = {}
    reg_predictions = {}
    
    for username in test_usernames:
        # Extract features
        features = extract_enhanced_features(username)
        
        # Make predictions
        category = predict_category(features)
        engagement = predict_engagement(features)
        
        cls_predictions[username] = category
        reg_predictions[username] = engagement
    
    # Save predictions
    save_predictions(cls_predictions, 'prediction-classification-round3.json')
    save_predictions(reg_predictions, 'prediction-regression-round3.json')
    
    print("\nPredictions saved to:")
    print("- prediction-classification-round3.json")
    print("- prediction-regression-round3.json")
    
    # Print statistics
    print("\nCategory distribution in predictions:")
    cat_dist = pd.Series(cls_predictions.values()).value_counts()
    for cat, count in cat_dist.items():
        print(f"{cat}: {count} ({count/len(cls_predictions)*100:.1f}%)")
    
    print("\nEngagement score statistics:")
    eng_scores = list(reg_predictions.values())
    print(f"Mean: {np.mean(eng_scores):.1f}")
    print(f"Median: {np.median(eng_scores):.1f}")
    print(f"Min: {np.min(eng_scores):.1f}")
    print(f"Max: {np.max(eng_scores):.1f}")

if __name__ == "__main__":
    main() 