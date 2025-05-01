from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F

def predict_sentiment(model_path, sentence):
    # Load the tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # Load the trained model
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set the model to evaluation mode
    
    # Tokenize the input text
    inputs = tokenizer(sentence, truncation=True, padding=True, return_tensors="pt")
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
    
    # Map prediction to sentiment label
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    sentiment = sentiment_map[predictions.item()]
    
    # Get probability scores for each class
    probs = probabilities[0].tolist()
    return sentiment, probs

# Specify your model path and test sentence here
MODEL_PATH = r"C:\Users\Mehmet\OneDrive\Desktop\TextMiningProject\sentiment_model"  # Path to the directory containing the model
#TEST_SENTENCE = "I hate this movie so much i disliked it and it was a waste of time"  # Replace with your actual test sentence
#TEST_SENTENCE = "I love this movie so much i liked it and it was not a waste of time"  # Replace with your actual test sentence
#TEST_SENTENCE = "this movie is about kid who is a good student and he lives in london but bad things happen to him"  # Replace with your actual test sentence

try:
    sentiment, probabilities = predict_sentiment(MODEL_PATH, TEST_SENTENCE)
    print(f"\nInput sentence: {TEST_SENTENCE}")
    print(f"Predicted Sentiment: {sentiment}")
    print("\nProbabilities for each class:")
    print(f"Negative: {probabilities[0]:.4f}")
    print(f"Neutral: {probabilities[1]:.4f}")
    print(f"Positive: {probabilities[2]:.4f}")
except Exception as e:
    print(f"Error occurred: {str(e)}")
