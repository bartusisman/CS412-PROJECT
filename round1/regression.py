import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import json
import os
import sys

def load_training_data():
    """Load training dataset with posts"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    data_path = os.path.join(data_dir, 'training-dataset.jsonl.gz')
    
    posts_data = []
    print(f"\nAttempting to read: {data_path}")
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if 'posts' in data:
                        for post in data['posts']:
                            # Handle None values by defaulting to 0
                            comment_count = post.get('comment_count')
                            like_count = post.get('like_count')
                            
                            post_features = {
                                'media_type': post.get('media_type', 'unknown'),
                                'comment_count': 0 if comment_count is None else int(comment_count),
                                'like_count': 0 if like_count is None else int(like_count),
                                'caption_length': len(str(post.get('caption', ''))),
                                'hashtag_count': len(post.get('hashtags', [])),
                                'mention_count': len(post.get('mentions', [])),
                            }
                            posts_data.append(post_features)
                except Exception as e:
                    continue
                
                if i % 1000 == 0:
                    print(f"Processed {i} lines... Found {len(posts_data)} valid posts", end='\r', flush=True)
    except Exception as e:
        print(f"\nError: Failed to process file: {str(e)}")
        sys.exit(1)
    
    if not posts_data:
        print("\nWarning: No posts found in training data!")
        sys.exit(1)
    
    print(f"\nSuccessfully loaded {len(posts_data)} posts")
    return pd.DataFrame(posts_data)

def load_test_data():
    """Load test posts for regression"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(current_dir, 'data', 'test-regression-round1.jsonl')
    
    test_data = []
    print(f"\nLoading test data from: {test_path}")
    
    try:
        with open(test_path, 'r') as f:
            for line in f:
                try:
                    post = json.loads(line.strip())
                    post_features = {
                        'id': post['id'],
                        'media_type': post.get('media_type', 'unknown'),
                        'comment_count': 0 if post.get('comment_count') is None else int(post.get('comment_count', 0)),
                        'caption_length': len(str(post.get('caption', ''))),
                        'hashtag_count': len(post.get('hashtags', [])),
                        'mention_count': len(post.get('mentions', [])),
                    }
                    test_data.append(post_features)
                except Exception as e:
                    print(f"Warning: Skipping invalid test data: {str(e)}")
                    continue
    except Exception as e:
        print(f"\nError: Failed to load test data: {str(e)}")
        sys.exit(1)
    
    print(f"Loaded {len(test_data)} test posts")
    return pd.DataFrame(test_data)

def main():
    print("Loading training data...")
    train_df = load_training_data()
    
    # Prepare features and target
    feature_cols = ['comment_count', 'caption_length', 'hashtag_count', 'mention_count']
    X = pd.get_dummies(train_df[['media_type'] + feature_cols], columns=['media_type'])
    y = train_df['like_count']
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X[feature_cols] = scaler.fit_transform(X[feature_cols])
    
    # Train model
    print("\nTraining random forest regressor...")
    model = RandomForestRegressor(
        n_estimators=1000,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    model.fit(X, y)
    
    # Print feature importance
    print("\nTop 10 Most Important Features:")
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance_df.head(10).iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    # Load and process test data
    print("\nProcessing test data...")
    test_df = load_test_data()
    test_X = pd.get_dummies(test_df[['media_type'] + feature_cols], columns=['media_type'])
    
    # Ensure test features match training features
    for col in X.columns:
        if col not in test_X.columns:
            test_X[col] = 0
    test_X = test_X[X.columns]
    
    # Scale test features
    test_X[feature_cols] = scaler.transform(test_X[feature_cols])
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(test_X)
    
    # Create predictions dictionary
    predictions_dict = {
        str(test_df['id'].iloc[i]): int(max(0, predictions[i]))
        for i in range(len(predictions))
    }
    
    # Save predictions
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'prediction-regression-round1.json'
    )
    
    with open(output_path, 'w') as f:
        json.dump(predictions_dict, f, indent=4)
    
    print(f"\nPredictions saved to: {output_path}")
    print(f"Number of predictions: {len(predictions_dict)}")
    print("\nSample predictions:")
    for post_id in list(predictions_dict.keys())[:5]:
        print(f"Post {post_id}: {predictions_dict[post_id]} likes")

if __name__ == "__main__":
    main() 