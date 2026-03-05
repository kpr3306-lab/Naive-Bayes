import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    file = request.files['file']
    target = request.form['target']
    train_size = float(request.form['train_size'])
    model_name = request.form['model']

    df = pd.read_csv(file)

    X = df.drop(target, axis=1)
    y = df[target]

    # Encode categorical features
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=42
    )

    if model_name == "Gaussian":
        model = GaussianNB()
    elif model_name == "Multinomial":
        model = MultinomialNB()
    else:
        model = BernoulliNB()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cm.png')
    plt.savefig(img_path)
    plt.close()

    return render_template('result.html', accuracy=round(acc,3), image='cm.png')

if __name__ == '__main__':
    app.run(debug=True)