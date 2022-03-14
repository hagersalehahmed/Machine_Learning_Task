# Machine-Learning-Task
## Many countries speak Arabic; however, each country has its own dialect, the aim of this task is to build a model that predicts the dialect given the text.

### This task includes many steps
1. Data fetching script
2. Data pre-procssing
3. Applying machnine learing (ML)
4. Appying deep learning (DL)
5. Deployment script

# Requirements
You must have Scikit Learn, Keras, Pandas KerasTuner and Flask (for API) installed.
 

# This project has four major parts :
    
1. Data fetching script.ipynb - This includes code to retrieve dataset using ids.
2. Data pre-processing script.ipynb- This includes some steps to clean text data such as remove URL, remove some special characters,  remove arabic stop words, romove emoji.
3. split dataset.ipynb - this includes code to split dataset using stratified method intto training set and testing set. Training set is used to optmize and train models and testing set is used to evaluate models. 

#Training and Evaluate models includes many files
### First approach: Six ML model is used support vector machine (SVM), Logistic Regression(LR), naive bayes (NB),  k-nearest neighbors(KNN), decision tree (DT) and random forest (RF)
1. ML with Count Vector and Optimzation Method.ipynb - This includes - 
 * CountVectorizer as featur extraction methods
 * Grid search with cross-validation is used to optimize models.
 * Result of appying ML models for traning set and testing set.
 
 2 ML with TF_IDF.ipynb.ipynb - This includes - 
 * TfidfVectorizer as featur extraction methods
 * Result of appying ML models for traning set and testing set.
 
 3. ML with Word2Vec.ipynb - This includes - 
 * Word2Vec is used to bulid word vectors as featur extraction methods
 * Result of appying ML models for traning set and testing set.

### Second approach: DL model is used LSTM and GRU with two pre-trained word embedding
Files
1. KerasTuner is used to optimize LSTM and GRU
https://keras.io/keras_tuner/
2. Twitter-CBOW and Twitter-SkipGram with 300Vec-Size as word embedding is used: https://github.com/bakrianoo/aravec
Four files LSTM, GRU files

# Deployment script

1. app.py - This contains Flask APIs that receives Text through GUI or API calls, computes the precited value based on our model and returns it.
2. request.py - This uses requests module to call APIs already defined in app.py and dispalys the returned value.

