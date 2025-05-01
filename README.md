# Movie Review Sentiment Analysis Project

## Overview
This project implements a text mining and sentiment analysis system for movie reviews from Letterboxd. It uses various natural language processing techniques including TF-IDF vectorization and Naive Bayes classification to analyze and categorize movie reviews into positive, neutral, and negative sentiments.

## Project Structure
- `LetterboxdScraper.py`: Web scraper for collecting movie reviews from Letterboxd
- `FilterEnglishComments.py`: Script to filter and retain English language comments
- `NormalizationCode.py`: Text preprocessing and normalization
- `TF-IDF_Vectorization.py`: Implementation of TF-IDF feature extraction
- `TFIDF_NaiveBayes.py`: Main classification model implementation
- `Optimization&GraphGeneration.py`: Model optimization and visualization tools

## Dataset Statistics
Current dataset composition:
- Positive Reviews: 198
- Neutral Reviews: 189
- Negative Reviews: 40
- Total Reviews: 427

## Model Performance
TF-IDF Features:
- Vector Size: (371, 2604)

Naive Bayes Classification Results:
```
              precision    recall  f1-score   support
    negative       0.50      0.73      0.59        30
     neutral       0.68      0.42      0.52        31
    positive       0.59      0.55      0.57        31

    accuracy                           0.57        92
   macro avg       0.59      0.57      0.56        92
weighted avg       0.59      0.57      0.56        92
```

## Setup and Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Installation
1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Required Dependencies
- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## Usage

1. To scrape new reviews:
```bash
python LetterboxdScraper.py
```

2. To filter English comments:
```bash
python FilterEnglishComments.py
```

3. To run the sentiment analysis:
```bash
python TFIDF_NaiveBayes.py
```

## Results
The model achieves:
- Overall accuracy: 57%
- Balanced performance across sentiment classes
- Strong performance in negative sentiment detection (73% recall)

## Future Improvements
- Increase dataset size, particularly for negative reviews
- Implement advanced feature engineering techniques
- Explore deep learning approaches
- Add cross-validation
- Implement ensemble methods

## License
[Specify your license here]

## Contributors
[Add contributor information here]
