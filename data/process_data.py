"""An ETL pipeline that reads the dataset, cleans it, and then stores it in a SQLite database."""

import pandas as pd
from sqlalchemy import create_engine

# load messages dataset
messages = pd.read_csv("messages.csv")

# load categories dataset
categories = pd.read_csv("categories.csv")

# remove duplicates
messages.drop_duplicates(inplace=True)
categories.drop_duplicates(inplace=True)

# merge datasets
df = pd.merge(messages, categories, on="id")

# create a dataframe of the 36 individual category columns
categories = categories["categories"].str.split(";", expand=True)

# select the first row of the categories dataframe
row = categories.iloc[0]

# use this row to extract a list of new column names for categories.
category_colnames = row.apply(lambda x: x[:-2])

# rename the columns of `categories`
categories.columns = category_colnames

# convert category values to just numbers 0 or 1
for column in categories:
    categories[column] = categories[column].str[-1]
    categories[column] = categories[column].astype(int)

# drop the original categories column from `df`
df.drop("categories", axis=1, inplace=True)

# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df, categories], axis=1)

# drop duplicates
df.drop_duplicates(inplace=True)

# save the clean dataset into an sqlite database
engine = create_engine('sqlite:///mydatabase.db')
df.to_sql('mytable', engine, index=False)
