"""Flask file that runs the Flask web app."""

import json

import joblib
import pandas as pd
import plotly
from flask import Flask
from flask import render_template, request
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine("sqlite:///../data/DisasterResponse.db")
df = pd.read_sql_table("messages_categories", engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():
    # extract data needed for visuals
    top_10_categories = df.iloc[:, 4:].sum().sort_values(ascending=False)[:10]
    categories_per_message = df.iloc[:, 4:].sum(axis=1).value_counts().sort_index()

    # create visuals
    graphs = [
        {
            "data": [Bar(x=top_10_categories.index, y=top_10_categories)],
            "layout": {
                "title": "Top 10 message categories",
                "yaxis": {"title": "Number of messages"},
                "xaxis": {"title": "Category"},
            },
        },
        {
            "data": [Bar(x=categories_per_message.index, y=categories_per_message)],
            "layout": {
                "title": "Number of categories per message",
                "yaxis": {"title": "Number of messages"},
                "xaxis": {"title": "Number of categories"},
            },
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
