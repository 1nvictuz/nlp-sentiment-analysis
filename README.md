# NLP Sentiment Analysis

## Overview
This AI Developer Boot Camp project uses Python to classify text reviews as positive or negative.

## What I Did
- Used the course CSV's `text` and `sentiment` columns.
- Split 20 reviews into training (80%) and testing (20%).
- Converted words into numerical features using CountVectorizer.
- Trained an SVM classifier and measured test accuracy.
- Compared the result with a majority-class baseline.

## Tools
Python 3.12, Pandas, Scikit-learn, VS Code.

## Run
```bash
pip install -r requirements.txt
python sentiment_analysis.py
```

## Results
Run the script to generate `results.txt` with the actual accuracy, baseline score, and package versions. Random seed: 42.

## Limitations
The dataset has only 20 reviews. Four test reviews are too few to judge real-world accuracy. The data came from the course; redistribution permission has not been verified.
