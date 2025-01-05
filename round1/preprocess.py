import pandas as pd
import numpy as np
import json
import os
import sys
import re
from sklearn.preprocessing import StandardScaler

def clean_username(username):
    """Clean username by removing extra text and newlines"""
    # Remove the entry number and newline
    username = re.sub(r'&entry\.[0-9]+=.*\n?', '', username)
    return username.lower().strip()

def get_category_keywords():
    """Define keywords for each category"""
    return {
        'Tech': ['tech', 'software', 'mobile', 'digital', 'robot', 'yazilim', 'bilisim', 'telekom'],
        'Food': ['food', 'chef', 'kitchen', 'cook', 'yemek', 'restaurant', 'cafe', 'mutfak', 'sut', 'cikolata'],
        'Sports': ['sport', 'fitness', 'futbol', 'spor', 'gym', 'athletic', 'basketbol', 'federasyon'],
        'Gaming': ['game', 'gaming', 'player', 'oyun', 'gamer'],
        'Fashion': ['fashion', 'style', 'design', 'moda', 'jewellery', 'collection'],
        'Entertainment': ['entertainment', 'music', 'show', 'dans', 'tiyatro', 'muzik', 'oyuncu', 'artist'],
        'Travel': ['travel', 'tour', 'hotel', 'tatil', 'gezi', 'seyahat'],
        'Art': ['art', 'artist', 'design', 'sanat', 'muzesi', 'galeri'],
        'Health and Lifestyle': ['health', 'life', 'saglik', 'yasam', 'diet', 'wellness', 'medical'],
        'Mom and Children': ['mom', 'baby', 'child', 'anne', 'bebek', 'cocuk', 'kids']
    }

def extract_username_features(username):
    """Extract features from username"""
    # Clean the username first
    username = clean_username(username)
    
    # Basic features
    features = {
        'username_length': len(username),
        'has_numbers': int(bool(re.search(r'\d', username))),
        'has_underscore': int('_' in username),
        'has_dot': int('.' in username),
        'word_count': len(username.split('_')),
        'starts_with_letter': int(bool(re.match(r'^[a-zA-Z]', username))),
        'ends_with_number': int(bool(re.search(r'\d$', username))),
        'capital_letter_count': sum(1 for c in username if c.isupper()),
        'number_count': sum(1 for c in username if c.isdigit()),
        'letter_count': sum(1 for c in username if c.isalpha())
    }
    
    # Add keyword-based features
    keywords = get_category_keywords()
    for category, words in keywords.items():
        category_score = 0
        for word in words:
            if word in username.lower():
                category_score += 1
        features[f'keyword_{category.lower().replace(" ", "_")}'] = category_score
    
    return features

def load_training_data():
    """Load training data"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level to project root
    data_dir = os.path.join(project_root, 'common_data')
    
    try:
        # Load training labels
        df = pd.read_csv(os.path.join(data_dir, 'train-classification.csv'))
        df.columns = ['username', 'category']
        
        # Clean usernames
        df['username'] = df['username'].apply(clean_username)
        
        # Standardize categories
        df['category'] = df['category'].apply(lambda x: x.strip())
        
        # Calculate category distribution
        category_dist = df['category'].value_counts()
        print("\nCategory Distribution in Training Data:")
        for cat, count in category_dist.items():
            print(f"{cat}: {count} ({count/len(df)*100:.1f}%)")
        
        # Extract features for each username
        features_list = []
        for username in df['username']:
            features_list.append(extract_username_features(username))
        
        # Convert to DataFrame
        features = pd.DataFrame(features_list, index=df['username'])
        
        # Scale numerical features
        scaler = StandardScaler()
        numerical_cols = ['username_length', 'word_count', 'capital_letter_count', 
                         'number_count', 'letter_count']
        features[numerical_cols] = scaler.fit_transform(features[numerical_cols])
        
        return features, df
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("\nPlease ensure the file exists and is readable:")
        print("- train-classification.csv")
        sys.exit(1)

def main():
    print("Loading training data...")
    features, labels = load_training_data()
    
    print("\nSaving processed data...")
    processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save features and labels
    features.to_csv(os.path.join(processed_dir, 'features.csv'))
    labels['category'].to_csv(os.path.join(processed_dir, 'labels.csv'))
    
    # Save category mapping
    username_categories = dict(zip(labels['username'], labels['category']))
    with open(os.path.join(processed_dir, 'category_mapping.json'), 'w') as f:
        json.dump(username_categories, f, indent=4)
    
    print("\nPreprocessing complete!")
    print(f"Features shape: {features.shape}")
    print(f"Number of features: {len(features.columns)}")
    print("\nFeature list:")
    for col in features.columns:
        print(f"- {col}: {features[col].mean():.2f} (mean)")

if __name__ == "__main__":
    main()