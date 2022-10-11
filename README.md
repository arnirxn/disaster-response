# disaster-response

A pipeline to classify disaster response messages

## Introduction

This project contains real messages sent during disaster events. A machine learning pipeline is created to categorize
these events so that messages can be sent to the appropriate disaster relief agency. The project includes a web app
where an emergency worker can input a new message and get classification results in several categories.

## Project Components

### 1. ETL Pipeline

A Python script, process_data.py, that:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

### 3. ML Pipeline
A Python script, train_classifier.py, that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

### 4. Web App

A Flask web app where a new message can be input and classification results in several categories are returned. The web
app also displays visualizations of the data.

## Instructions

### 1. Create a sqlite db with the processed data

python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

### 2. Train the model and save a pickle file of the model

python train_classifier.py ../data/DisasterResponse.db classifier.pkl

### 3. Deploy the web application locally

python run.py

## Files in the repository

- app
  - template
    - master.html # main page of web app
    - go.html # classification result page of web app
  - run.py # Flask file that runs app
- data
  - disaster_categories.csv # data to process
  - disaster_messages.csv # data to process
  - process_data.py
  - DisasterResponse.db # database to save clean data to
- models
  - train_classifier.py
  - classifier.pkl # saved model
- README.md
- requirements.txt # libraries to install to run this project


