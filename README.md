# Women's Clothing Review - NLP Sentiment Analysis

## Project Overview
This project focuses on **Natural Language Processing (NLP) and Sentiment Analysis** of women's clothing reviews. The goal is to analyze customer feedback, extract sentiment, and build predictive models to classify positive and negative reviews.

## Dataset
- The dataset consists of women's clothing reviews with features such as:
  - `Review Text`: Customer-written feedback.
  - `Rating`: A numerical score indicating customer satisfaction.
  - `Category & Subcategories`: Clothing types.
  - `Recommend Flag`: Whether the customer recommends the product.

## Project Workflow
### 1. **Data Preprocessing**
   - Handle missing values and remove unnecessary columns.
   - Convert categorical variables to numerical format.
   - Perform **text cleaning**, including:
     - Tokenization
     - Stopword removal
     - Stemming using **Porter Stemmer**
     - Expansion of contractions
     - Removing unnecessary characters

### 2. **Exploratory Data Analysis (EDA)**
   - Visualize **age distribution, category frequency**, and sentiment trends.
   - Compute **word count, polarity, and subjectivity scores** using `TextBlob`.
   - Generate **bar charts and histograms** to understand customer sentiments.

### 3. **Sentiment Analysis**
   - Use **VADER Sentiment Analyzer** to compute polarity scores.
   - Categorize reviews into **Positive, Negative, or Neutral**.
   - Create **visual representations** of sentiment distributions.

### 4. **Feature Engineering**
   - Extract features like:
     - Word Count, Character Count, Punctuation Count
     - Upper Case Word Count, Title Word Count
   - Convert `Review Text` into **TF-IDF & Count Vectorizer representations**.

### 5. **Model Building & Evaluation**
   - Train **Logistic Regression Classifier** to predict review sentiment.
   - Evaluate performance using **Confusion Matrix, Precision, Recall, F1-score**.
   - Improve model accuracy with **N-grams and TF-IDF transformations**.

## Installation & Dependencies
To run this project, install the required Python libraries:

```bash
pip install numpy pandas nltk seaborn matplotlib textblob scikit-learn contractions
```

## Usage
Clone the repository and run the Jupyter Notebook:

```bash
git clone https://github.com/anouskap/NLP.git
cd Women's clothing review- NLP
jupyter notebook
```

Then, open **"Women's clothing review- NLP.ipynb"** in Jupyter Notebook and execute the cells sequentially.

## Results & Findings
- **Sentiment Analysis** showed a higher number of **positive reviews**.
- The **Logistic Regression model** achieved **good accuracy** in predicting review sentiment.
- **NLP feature extraction** significantly improved classification performance.

## Future Work
- Experiment with **LSTMs and Transformer models** for better NLP analysis.
- Incorporate **aspect-based sentiment analysis** to determine feature-specific sentiments.
- Improve model performance using **word embeddings (Word2Vec, GloVe, or BERT)**.

## Contributors
- **Anouska Priya**
