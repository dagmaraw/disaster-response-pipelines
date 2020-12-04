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
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM {}".format(database_filepath[5:-3]),engine)
    X = df.message.values
    Y = df.iloc[:,4:]
    category_names = Y.columns.values
    
    return X,Y,category_names


def tokenize(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    text = re.sub(r"[^a-zA-Z0-9]"," ",text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens

def build_model():
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(AdaBoostClassifier(learning_rate=0.5,n_estimators=200)))
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    
    #metrics_summary = pd.DataFrame(index = category_names, columns = ['accuracy', 'precision', 'recall', 'f-1_score'])
    #for i, c in enumerate(category_names): 
    #    metrics_summary.loc[c,'accuracy'] = accuracy_score(Y_test.iloc[:,i],y_pred[:,i])
    #    metrics_summary.loc[c,'precision'] = precision_score(Y_test.iloc[:,i],y_pred[:,i])
    #    metrics_summary.loc[c,'recall'] = recall_score(Y_test.iloc[:,i],y_pred[:,i])
    #    metrics_summary.loc[c,'f-1_score'] = fbeta_score(Y_test.iloc[:,i],y_pred[:,i],beta=1)

    #metrics_summary.loc['average'] = metrics_summary.mean(axis=0)
    #print(metrics_summary)
    
    for i, c in enumerate(category_names): 
        print(c)
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))

def save_model(model, model_filepath):
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