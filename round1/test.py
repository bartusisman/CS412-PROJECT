import pandas as pd
import joblib
import os
import json

def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models', 'classifier.joblib')
    return joblib.load(model_path)

def predict_single_account(model, is_influencer, has_mention):
    features = pd.DataFrame([[is_influencer, has_mention]], 
                          columns=['is_influencer', 'has_mention'])
    prediction = model.predict(features)[0]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    category_mapping = pd.read_csv(os.path.join(current_dir, 'processed', 'category_mapping.csv'), 
                                 index_col=0)['0'].to_dict()
    
    return category_mapping[prediction]

def main():
    # Load the original data to get usernames
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'Annotated Users CS412.csv')
    df = pd.read_csv(data_path)
    
    model = load_model()
    
    # Create predictions dictionary
    predictions = {}
    for _, row in df.iterrows():
        is_influencer = 1 if row['accountType'] == 'Influencer' else 0
        has_mention = 1 if row['influencerMention'] == 'Yes' else 0
        username = row['url'].split('/')[-1]  # Extract username from URL
        
        prediction = predict_single_account(model, is_influencer, has_mention)
        predictions[username] = prediction
    
    # Save predictions to JSON file
    output_path = os.path.join(current_dir, 'prediction-classification-round1.json')
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=4)
    
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main() 