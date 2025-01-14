import pandas as pd
import numpy as np
import json
import os
import re
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import langdetect
from collections import Counter
from langdetect import detect
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

def clean_username(username):
    """Clean username by removing extra text and newlines"""
    username = re.sub(r'&entry\.[0-9]+=.*\n?', '', username)
    return username.lower().strip()

def detect_language(text):
    """Detect if text contains Turkish characters"""
    turkish_chars = set('çğıöşüÇĞİÖŞÜ')
    return 1 if any(c in turkish_chars for c in text) else 0

def get_enhanced_category_keywords():
    """Define enhanced keywords for each category with both English and Turkish terms"""
    return {
        'Tech': [
            'tech', 'software', 'mobile', 'digital', 'robot', 'yazilim', 'bilisim', 'telekom',
            'bilgisayar', 'teknoloji', 'web', 'app', 'uygulama', 'sistem', 'network', 'ag',
            'cloud', 'bulut', 'data', 'veri', 'yapay', 'zeka', 'ai', 'ml', 'siber', 'dev',
            'developer', 'code', 'coder', 'coding', 'programmer', 'programming', 'backend',
            'frontend', 'fullstack', 'engineer', 'engineering', 'cyber', 'security', 'hack',
            'hacker', 'linux', 'windows', 'apple', 'android', 'ios', 'mobile', 'web', 'net',
            'java', 'python', 'ruby', 'php', 'html', 'css', 'js', 'javascript', 'sql', 'database'
        ],
        'Food': [
            'food', 'chef', 'kitchen', 'cook', 'yemek', 'restaurant', 'cafe', 'mutfak',
            'sut', 'cikolata', 'restoran', 'lokanta', 'gurme', 'gourmet', 'pizza', 'burger',
            'kebap', 'baklava', 'pasta', 'tatli', 'sweet', 'coffee', 'kahve', 'chef', 'asci',
            'foodie', 'cuisine', 'culinary', 'bake', 'baker', 'bakery', 'pastry', 'cake',
            'dessert', 'chocolate', 'ice', 'cream', 'organic', 'vegan', 'vegetarian', 'meat',
            'fish', 'seafood', 'kitchen', 'cook', 'cooking', 'recipe', 'recipes', 'meal',
            'breakfast', 'lunch', 'dinner', 'snack', 'drink', 'drinks', 'bar', 'pub'
        ],
        'Sports': [
            'sport', 'fitness', 'futbol', 'spor', 'gym', 'athletic', 'basketbol', 'federasyon',
            'voleybol', 'volleyball', 'football', 'basketball', 'tennis', 'tenis', 'run',
            'kosu', 'marathon', 'maraton', 'fitness', 'antrenman', 'training', 'coach',
            'team', 'club', 'player', 'game', 'match', 'championship', 'league', 'tournament',
            'cup', 'olympic', 'olympics', 'medal', 'winner', 'champion', 'athlete', 'athletic',
            'sports', 'workout', 'exercise', 'train', 'trainer', 'training', 'muscle', 'gym',
            'fitness', 'fit', 'running', 'runner', 'swim', 'swimming', 'cyclist', 'cycling'
        ],
        'Gaming': [
            'game', 'gaming', 'player', 'oyun', 'gamer', 'steam', 'twitch', 'esport',
            'playstation', 'xbox', 'nintendo', 'gaming', 'streamer', 'stream', 'play',
            'pro', 'professional', 'team', 'clan', 'guild', 'esports', 'tournament',
            'champion', 'championship', 'competitive', 'console', 'pc', 'mobile', 'game',
            'gaming', 'gamer', 'play', 'player', 'playing', 'stream', 'streamer', 'streaming',
            'youtube', 'youtuber', 'content', 'creator', 'video', 'videos', 'live', 'twitch'
        ],
        'Fashion': [
            'fashion', 'style', 'design', 'moda', 'jewellery', 'collection', 'butik',
            'boutique', 'accessories', 'aksesuar', 'clothes', 'giyim', 'kiyafet', 'shoes',
            'ayakkabi', 'textile', 'tekstil', 'brand', 'marka', 'designer', 'tasarim',
            'trend', 'trendy', 'chic', 'elegant', 'luxury', 'luxe', 'premium', 'vintage',
            'retro', 'modern', 'contemporary', 'classic', 'minimal', 'maximalist', 'boho',
            'bohemian', 'street', 'streetwear', 'urban', 'casual', 'formal', 'dress',
            'dresses', 'wear', 'wearing', 'outfit', 'outfits', 'look', 'looks', 'style'
        ],
        'Entertainment': [
            'entertainment', 'music', 'show', 'dans', 'tiyatro', 'muzik', 'oyuncu', 'artist',
            'eglence', 'konser', 'concert', 'festival', 'sahne', 'stage', 'performans',
            'performance', 'sinema', 'cinema', 'film', 'movie', 'dizi', 'series', 'tv',
            'television', 'radio', 'broadcast', 'broadcasting', 'media', 'social', 'content',
            'creator', 'creative', 'art', 'artist', 'artistic', 'perform', 'performer',
            'performing', 'dance', 'dancer', 'dancing', 'sing', 'singer', 'singing', 'band',
            'musician', 'musical', 'comedy', 'comedian', 'funny', 'humor', 'entertainment'
        ],
        'Travel': [
            'travel', 'tour', 'hotel', 'tatil', 'gezi', 'seyahat', 'turizm', 'tourism',
            'holiday', 'vacation', 'trip', 'otel', 'resort', 'beach', 'plaj', 'deniz',
            'sea', 'mountain', 'dag', 'camping', 'kamp', 'outdoor', 'adventure', 'macera',
            'explore', 'explorer', 'exploring', 'wander', 'wanderlust', 'traveler',
            'travelling', 'backpack', 'backpacker', 'nomad', 'digital', 'world', 'global',
            'international', 'local', 'guide', 'tour', 'tourist', 'tourism', 'destination',
            'locations', 'places', 'cities', 'countries', 'culture', 'cultural'
        ],
        'Art': [
            'art', 'artist', 'design', 'sanat', 'muzesi', 'galeri', 'exhibition', 'sergi',
            'paint', 'resim', 'heykel', 'sculpture', 'ceramic', 'seramik', 'creative',
            'yaratici', 'workshop', 'atolye', 'craft', 'zanaat', 'photography', 'fotograf',
            'photo', 'photographer', 'photoshoot', 'portrait', 'landscape', 'nature',
            'digital', 'traditional', 'contemporary', 'modern', 'abstract', 'realistic',
            'surreal', 'minimalist', 'maximalist', 'illustration', 'illustrator', 'draw',
            'drawing', 'sketch', 'sketching', 'paint', 'painting', 'painter', 'gallery'
        ],
        'Health and Lifestyle': [
            'health', 'life', 'saglik', 'yasam', 'diet', 'wellness', 'medical', 'hospital',
            'hastane', 'doctor', 'doktor', 'clinic', 'klinik', 'healthy', 'saglikli',
            'fitness', 'yoga', 'meditation', 'meditasyon', 'organic', 'organik', 'natural',
            'wellness', 'wellbeing', 'holistic', 'alternative', 'medicine', 'therapy',
            'therapeutic', 'healing', 'healer', 'mental', 'physical', 'body', 'mind',
            'spirit', 'spiritual', 'nutrition', 'nutritionist', 'diet', 'dietary',
            'lifestyle', 'living', 'life', 'coach', 'coaching', 'consultant', 'advisor'
        ]
    }

def extract_enhanced_features(username, profile_data=None):
    """Extract enhanced features from username and profile data"""
    features = {}
    username_lower = username.lower()
    
    # 1. Basic features from username
    features['length'] = len(username)
    features['word_count'] = len(username.split())
    features['digit_count'] = sum(c.isdigit() for c in username)
    features['alpha_count'] = sum(c.isalpha() for c in username)
    features['upper_count'] = sum(c.isupper() for c in username)
    features['special_count'] = len(re.findall(r'[^a-zA-Z0-9]', username))
    
    # 2. Pattern features from username
    features['has_number'] = int(bool(re.search(r'\d', username)))
    features['has_special'] = int(bool(re.search(r'[^a-zA-Z0-9]', username)))
    features['has_year'] = int(bool(re.search(r'(19|20)\d{2}', username)))
    features['is_turkish'] = detect_language(username)
    
    # 3. Category-specific pattern detection from username
    features['has_gaming'] = int(bool(re.search(r'(game|gaming|gamer|play|steam|twitch)', username_lower)))
    features['has_tech'] = int(bool(re.search(r'(tech|dev|code|prog|web)', username_lower)))
    features['has_food'] = int(bool(re.search(r'(food|chef|cook|kitchen|cafe)', username_lower)))
    features['has_sports'] = int(bool(re.search(r'(sport|fitness|gym|run|coach)', username_lower)))
    features['has_fashion'] = int(bool(re.search(r'(fashion|style|moda|design)', username_lower)))
    features['has_art'] = int(bool(re.search(r'(art|artist|design|photo|creative)', username_lower)))
    features['has_travel'] = int(bool(re.search(r'(travel|tour|trip|wanderlust)', username_lower)))
    features['has_health'] = int(bool(re.search(r'(health|wellness|saglik|fit)', username_lower)))
    features['has_mom'] = int(bool(re.search(r'(mom|baby|anne|bebek|child)', username_lower)))
    features['has_entertainment'] = int(bool(re.search(r'(music|film|show|artist|band)', username_lower)))
    
    # 4. Get category keywords
    keywords = get_enhanced_category_keywords()
    
    # Add keyword-based features
    for category, words in keywords.items():
        category_lower = category.lower().replace(' and ', '_').replace(' ', '_')
        count = 0
        for word in words:
            if word.lower() in username_lower:
                count += 1
        features[f'keyword_{category_lower}'] = count
        
    # 5. Add confidence scores based on keyword matches
    for category in keywords.keys():
        category_lower = category.lower().replace(' and ', '_').replace(' ', '_')
        features[f'confidence_{category_lower}'] = features[f'keyword_{category_lower}'] / len(keywords[category])
    
    # 6. Add profile data features if available
    if profile_data:
        # Add follower and following counts
        features['followers_count'] = profile_data.get('followers_count', 0)
        features['following_count'] = profile_data.get('following_count', 0)
        features['follower_following_ratio'] = features['followers_count'] / (features['following_count'] + 1)
        
        # Add tweet metrics
        features['tweets_count'] = profile_data.get('tweets_count', 0)
        features['tweets_per_day'] = features['tweets_count'] / (profile_data.get('account_age_days', 1) + 1)
        
        # Add engagement metrics
        features['likes_count'] = profile_data.get('likes_count', 0)
        features['likes_per_tweet'] = features['likes_count'] / (features['tweets_count'] + 1)
        
        # Add profile completeness features
        features['has_profile_image'] = int(profile_data.get('has_profile_image', False))
        features['has_banner_image'] = int(profile_data.get('has_banner_image', False))
        features['has_description'] = int(bool(profile_data.get('description', '')))
        features['has_location'] = int(bool(profile_data.get('location', '')))
        features['has_url'] = int(bool(profile_data.get('url', '')))
        
        # Add text-based features from description
        if profile_data.get('description'):
            desc_lower = profile_data['description'].lower()
            for category, words in keywords.items():
                category_lower = category.lower().replace(' and ', '_').replace(' ', '_')
                count = 0
                for word in words:
                    if word.lower() in desc_lower:
                        count += 1
                features[f'desc_keyword_{category_lower}'] = count
                features[f'desc_confidence_{category_lower}'] = count / len(keywords[category])
    
    return features

def process_test_data(test_file):
    """Process test data"""
    # Load test data
    test_data = []
    with open(test_file, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    # Process test data
    test_features_list = []
    test_usernames = []
    for item in test_data:
        username = item['username']
        profile_data = item.get('profile_data', {})
        features = extract_enhanced_features(username, profile_data)
        test_features_list.append(features)
        test_usernames.append(username)
    
    # Convert to DataFrame
    test_features = pd.DataFrame(test_features_list, index=test_usernames)
    
    # Ensure test features match training features
    processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')
    train_features = pd.read_csv(os.path.join(processed_dir, 'features.csv'), index_col=0)
    
    # Create a DataFrame with zeros for missing columns
    missing_cols = set(train_features.columns) - set(test_features.columns)
    if missing_cols:
        missing_data = pd.DataFrame(0, index=test_features.index, columns=list(missing_cols))
        test_features = pd.concat([test_features, missing_data], axis=1)
    
    # Reorder columns to match training data
    test_features = test_features[train_features.columns]
    
    return test_features, test_usernames

def save_predictions(predictions, output_file):
    """Save predictions to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=4)

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

def load_and_process_data():
    """Load and process training data with enhanced features"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(current_dir, 'data', 'Test Classification Round 3.dat')
    
    # Load test usernames
    with open(test_file, 'r') as f:
        usernames = [line.strip() for line in f]
    
    # Process data
    features_list = []
    
    for username in usernames:
        features = extract_enhanced_features(username)
        features_list.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list, index=usernames)
    
    # Create processed directory if it doesn't exist
    processed_dir = os.path.join(current_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save processed data
    features_df.to_csv(os.path.join(processed_dir, 'features.csv'))
    
    print("\nFeatures shape:", features_df.shape)
    print("Number of test samples:", len(usernames))
    
    return features_df

def main():
    """Main function"""
    print("Loading and processing data...")
    features = load_and_process_data()
    
    print("\nFeatures shape:", features.shape)
    print("Sample features:", list(features.columns)[:10])

if __name__ == "__main__":
    main() 