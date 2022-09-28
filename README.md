# disaster-response
A pipeline to classify disaster response messages

# Project Components
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
A Flask web app where a new message can be input and classification results in several categories are returned. 
The web app also displays visualizations of the data.

