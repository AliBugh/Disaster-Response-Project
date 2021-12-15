import sys
# import libraries
import pandas as pd
import re
import pickle
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
nltk.download(['punkt','stopwords' , 'wordnet' , 'averaged_perceptron_tagger'])
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath, echo=True)
    df =  pd.read_sql_table('DisasterResponseTA',engine)
    X = df['message']
    Y = df.drop(columns=['id','message','original','genre'])
    category_names = Y.columns
    return X ,Y , category_names


def tokenize(text):
    # Convert to lowercase
    text = text.lower() 
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    # tokenize text
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
     pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))
                ])
    
     parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__min_samples_split': [2, 4]
    }
    
     cv = GridSearchCV(pipeline,parameters)

    
     return cv


def evaluate_model(model, X_test, Y_test, category_names):
     y_pred = model.predict(X_test)
     y_pred = pd.DataFrame(y_pred,columns= category_names)
     for col in category_names:
         print('Column Name: ',col)
         print(classification_report(Y_test[col], y_pred[col]))

def save_model(model, model_filepath):
     pickle.dump(model, open(model_filepath, 'wb'))


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