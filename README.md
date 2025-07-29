# 📧 Spam Email Classifier

A machine learning project that classifies emails as spam or ham (legitimate) using Natural Language Processing and the Naive Bayes algorithm.

## 🎯 Project Overview

This project implements a spam email classifier using scikit-learn's Multinomial Naive Bayes algorithm with TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. The model is trained to distinguish between spam and legitimate emails based on email content.

## 📊 Dataset

The project uses the **emails.csv** dataset which contains:
- Email content represented as word frequency features
- Binary classification labels (spam/ham)
- Over 3,000+ features representing different words
- Preprocessed email data ready for machine learning

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning library
  - `TfidfVectorizer` - Text feature extraction
  - `MultinomialNB` - Naive Bayes classifier
  - `train_test_split` - Data splitting
  - `accuracy_score`, `classification_report`, `confusion_matrix` - Model evaluation

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas scikit-learn numpy
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Mostafa-Faisal/ML-project-4-Spam-Email-Classifier.git
cd ML-project-4-Spam-Email-Classifier
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:
```bash
jupyter notebook Task2_SpamEmailClassifier.ipynb
```

## 📋 Project Structure

```
ML-project-4-Spam-Email-Classifier/
├── Task2_SpamEmailClassifier.ipynb    # Main notebook with implementation
├── emails.csv                         # Dataset with email features
└── README.md                          # Project documentation
```

## 🔍 Implementation Details

### 1. Data Loading and Preprocessing
- Load the dataset using pandas
- Check for missing values and data quality
- Convert categorical labels to binary (spam=1, ham=0)

### 2. Feature Engineering
- Use TF-IDF vectorization to convert text to numerical features
- Transform email content into feature vectors

### 3. Model Training
- Split data into training and testing sets (80/20)
- Train Multinomial Naive Bayes classifier
- Optimize model parameters

### 4. Model Evaluation
- Calculate accuracy score
- Generate classification report (precision, recall, F1-score)
- Create confusion matrix for detailed analysis

### 5. Prediction
- Test model on new email samples
- Classify emails as spam (1) or ham (0)

## 📈 Model Performance

The model achieves strong performance metrics:
- **Algorithm**: Multinomial Naive Bayes
- **Feature Extraction**: TF-IDF Vectorization
- **Train/Test Split**: 80/20
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score

## 💡 Usage Example

```python
# Example: Predict if new emails are spam
new_emails = [
    "Congratulations! You've won a free ticket.",  # Likely spam
    "Hi, can we meet tomorrow for lunch?"          # Likely ham
]

# Transform and predict
new_emails_tfidf = vectorizer.transform(new_emails)
predictions = model.predict(new_emails_tfidf)
print(predictions)  # Output: [1, 0] where 1=spam, 0=ham
```

## 🔧 Key Features

- ✅ **Text Preprocessing**: Handles email content cleaning
- ✅ **TF-IDF Vectorization**: Converts text to numerical features
- ✅ **Naive Bayes Classification**: Probabilistic spam detection
- ✅ **Model Evaluation**: Comprehensive performance metrics
- ✅ **Prediction Interface**: Easy-to-use prediction for new emails

## 📊 Results

The spam classifier demonstrates effective performance in distinguishing between spam and legitimate emails. Detailed results including accuracy scores, confusion matrix, and classification reports are available in the notebook.

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Mostafa Faisal**
- GitHub: [@Mostafa-Faisal](https://github.com/Mostafa-Faisal)

## 🙏 Acknowledgments

- Scikit-learn team for the excellent machine learning library
- The open-source community for dataset and tools
- Email spam detection research community

---

⭐ **Don't forget to star this repository if you found it helpful!**