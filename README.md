# Twitter Sentiment Classification Project

## Overview

This project implements a binary sentiment classification system for Twitter data.
The task is to classify tweets as positive (+1) or negative (-1) using GloVe word
embeddings and multiple machine learning classifiers.

## Dataset

- Training data: approximately 2 million tweets
- Validation data: 10 percent split from training data
- Test data: 10,000 unlabeled tweets
- Labels: -1 for negative sentiment, +1 for positive sentiment
- Source: Kaggle Twitter Sentiment Analysis dataset

Due to file size and license restrictions, the original dataset is not included
in this repository.

## Project Structure

```
project_text_classification/
  - run.py
  - main.py
  - requirements.txt
  - README.md

  - src/
      - baseline_classifier.py
      - tfidf_classifier.py
      - svm_classifier.py
      - negation_classifier.py
      - glove_template.py

  - scripts/
      - split_data.py
      - build_vocab.py
      - cut_vocab.py
      - pickle_vocab.py
      - cooc.py
      - build_vocab.sh
      - cut_vocab.sh

  - output/
      - (empty directory, generated at runtime)

  - public_kaggle_files/
      - twitter-datasets/
          - (dataset files go here)

  - report/
```

## Installation

### Requirements

- Python 3.8 or higher
- pip

### Dependency Installation

```
pip install -r requirements.txt
```

## Dependencies

- numpy 1.23.5
- scikit-learn 1.3.0
- pandas 1.5.3
- scipy 1.11.2
- matplotlib 3.7.2 (optional)

## Data Preparation

Download the Kaggle Twitter dataset and place the files in:

```
public_kaggle_files/twitter-datasets/
```

Required files:

- train_pos.txt
- train_neg.txt
- test_data.txt
- train_pos_full.txt
- train_neg_full.txt
- sample_submission.csv

The directories `output/` and `public_kaggle_files/` must exist before execution
but can be empty.

## Data Processing Pipeline

Stage 1: Vocabulary construction

```
cd scripts
python build_vocab.py
python cut_vocab.py
python pickle_vocab.py
```

Stage 2: Train and validation split

```
python split_data.py
```

This step must be executed before GloVe training to avoid data leakage.

Stage 3: Co-occurrence matrix computation

```
python cooc.py
```

Stage 4: GloVe embedding training

```
cd ../src
python glove_template.py
```

## Feature Representation Methods

Baseline method:
Mean pooling of word embeddings.

TF-IDF weighted embeddings:
Word vectors are weighted by TF-IDF scores before averaging.

Linear SVM:
Support Vector Machine trained on mean pooled embeddings.

Negation-aware embeddings:
Word embeddings within a fixed window after negation words have their signs flipped.

## Running the Project

Run the full pipeline:

```
python run.py
```

Step-by-step execution:

```
python main.py
```

Run individual classifiers:

```
cd src
python baseline_classifier.py
python tfidf_classifier.py
python svm_classifier.py
python negation_classifier.py
```

## Output

Final predictions are written to:

```
output/submission.csv
```

File format:

```
id,prediction
1,-1
2,1
...
10000,1
```

## Reproducibility

To reproduce the results:

1. Place the dataset files in `public_kaggle_files/twitter-datasets/`
2. Install dependencies using `pip install -r requirements.txt`
3. Run the project from the root directory using `python run.py`
4. Verify that `output/submission.csv` is generated

## License

This project is for educational purposes only.