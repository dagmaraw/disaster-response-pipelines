import sys

import nltk
nltk.download('punkt')
import pandas as pd
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet') # download for lemmatization
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, fbeta_score, recall_score, classification_report, accuracy_score, precision_score, make_scorer, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
import pickle
import numpy as np

def load_data(database_filepath):
    '''
    INPUT:
    database_filepath - (string) the filepath to the database with data

    OUTPUT:
    X - (numpy array) an array of messages
    Y - (pandas dataframe) a dataframe with classifications corresponding to each message
    category_names - (numpy array) an array of categories (columns from Y)

    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM {}".format(database_filepath[5:-3]),engine)
    X = df.message.values
    Y = df.iloc[:,4:]
    category_names = Y.columns.values
    
    return X,Y,category_names


def tokenize(text):
    '''
    INPUT:
    text - (string) text from the messages

    OUTPUT:
    tokens - (list) tokenized text

    Notes:
    Used in CountVectorizer in Pipeline in build_model function

    '''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    text = re.sub(r"[^a-zA-Z0-9]"," ",text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens

def build_model():
    '''
    OUTPUT:
    pipeline - (sklearn pipeline) pipeline of CountVectorizer, TFIDF transformer, and mulit-output classifier

    '''
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(AdaBoostClassifier(learning_rate=0.5,n_estimators=200)))
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    model - (pipeline) model created in build_model
    X_test - (numpy array) the testing set for x
    Y_test - (pandas dataframe) the testing set for y
    category_names - (numpy array) an array of categories (columns of Y)

    Description:
    Predicts classification labels for the test set. Prints classification_report for each category,
    which gives accuracy, recall, f-1 score, and precision, as well as averages

    '''
    y_pred = model.predict(X_test)
    
    for i, c in enumerate(category_names): 
        print(c)
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))

def save_model(model, model_filepath):
    '''
    INPUT:
    model - (pipeline) model created in build_model
    model_filepath - (string) file path for model

    OUTPUT:
    recs - (list) a list of recommendations for the user

    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()