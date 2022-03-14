# Machine-Learning-Task
## Many countries speak Arabic; however, each country has its own dialect; the aim of this task is to build a model that predicts the dialect given the text.

### This task includes many steps
1. Data fetching script
2. Data pre-processing
3. Applying machine learning (ML)
4. Applying deep learning (DL)
5. Deployment script

#Evaluating Models
The models are evaluated using four methods: accuracy,
precision , recall, and F-score foreach classes

# Requirements
You must have Scikit Learn, Keras, Pandas KerasTuner and Flask (for API) installed.
 

# This project has four major parts :
    
1. Data fetching script.ipynb - This includes code to retrieve text using ids.
2. Data pre-processing script.ipynb- This includes some steps to clean text data such as removing URL, removing some special characters, removing Arabic stop words, remove emoji.
3. split dataset.ipynb - This includes code to split dataset using stratified method into a training set and testing set. The training set is used to optimize, and train models and the testing set is used to evaluate models. 

#Training and Evaluate models include many files
### First approach: Six ML model is used support vector machine (SVM), Logistic Regression(LR), naive Bayes (NB),  k-nearest neighbors(KNN), decision tree (DT) and random forest (RF)
1. ML with Count Vector and Optimization Method.ipynb - This includes - 
 * CountVectorizer as feature extraction methods
 * Grid search with cross-validation is used to optimize models.
 * Result of applying ML models for the training set and testing set.
 
 2 ML with TF_IDF.ipynb.ipynb - This includes - 
 * TfidfVectorizer as featur extraction methods
 * Result of appying ML models for traning set and testing set.
 
 3. ML with Word2Vec.ipynb - This includes - 
 * Word2Vec is used to build word vectors as feature extraction methods
 * Result of applying ML models for the training set and testing set.

### Second approach: DL model is used LSTM and GRU with two pre-trained word embedding
Files
1. KerasTuner is used to optimize LSTM and GRU
https://keras.io/keras_tuner/
2. Twitter-CBOW and Twitter-SkipGram with 300Vec-Size as word embedding is used: https://github.com/bakrianoo/aravec
Four files LSTM, GRU files

# Deployment script

1. app.py - This contains Flask APIs that receive Text through GUI or API calls, compute the precited value based on our model, and return it.

