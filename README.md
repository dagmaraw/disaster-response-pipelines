### Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Motivation](#motivation)
4. [File Descriptions](#files)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

## Usage <a name="usage"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database:
	
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves:
	
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/

## Project Motivation<a name="motivation"></a>

The goal of this project was to analyze disaster data to build a model for an API that classifies disaster messages.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

## File Descriptions <a name="files"></a>

The main python scripts needed to run the web app are data/process_data.py, which cleans the data and stores it in a database,
and models/train_classifier.py, which trains a classifier and saves it. The data folder also contains the disaster_messages.csv and disaster_categories.csv data files.

The app folder contains run.py, which runs the web app.

Other files in the base directory are used for the 2 jupyter notebooks that were used as a guide to build process_data.py and train_classifier.py.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credit to Udacity and Figure Eight for the data. This project was part of Udacity's Data Scientist Nanodegree program.