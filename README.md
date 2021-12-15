# Disaster Response Pipeline

------
1. [Project Overview](#ProjectOverview)
2. [Installation](#installation)
3. [Project Components](#ProjectComponents)
4. [Results](#results)
5. [Files](#files)
6. [Acknowledgements](#acknowledgements)

## 1. Project Overview <a name="ProjectOverview"></a> 
Data set containing real messages that were sent during disaster events collected by [Figure Eight](https://www.figure-eight.com/) contains 30,000 messages, It has been encoded with 36 different categories.Creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency. 

## 2. Installation <a name="installation"></a>

- Python versions 3.*.
- Python Libraries:
    - Pandas.
    - Scikit-learn.
    - numpy.
    - nltk.
    - sqlalchemy.
  
## 3. Project Components <a name="ProjectComponents"></a> 
There are three main folder in this project :
1. **ETL Pipeline:** 
data/process_data.py, contains data cleaning pipeline that:
    - Loads the messages and categories datasets
    - Merges the two datasets
    - Cleans the data
    - Stores it in a SQLite database
        
2. **ML Pipeline:** 
models/train_classifier.py contains machine learning pipeline that:
    - Loads data from the SQLite database
    - Splits the dataset into training and test sets
    - Builds a text processing and machine learning pipeline
    - Trains and tunes a model using GridSearchCV
    - Outputs results on the test set
    - Exports the final model as a pickle file

3. **Flask Web App:** 
contains web app allow useres to enter messages and show the classified model results in real time.

## 4. Results <a name="results"></a> 
Screenshots of the web app.



## 5. Files <a name="files"></a>
<pre>
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
</pre>

## 6. Acknowledgements <a name="acknowledgements"></a> 
I would thank [Udacity](https://www.udacity.com/) for advice and providing great Data Science Nanodegree Program.
Also [Figure Eight](https://www.figure-eight.com/) for provide data sets to practice and learn about Data Science
