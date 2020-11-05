import os
import argparse
import subprocess
import logging
from flask import Flask, request, jsonify, render_template, url_for, flash, redirect
from werkzeug.exceptions import abort
from flask_cors import CORS
from baselines import load
import signal
import datetime
import sys
import shlex
import pymysql

argparser = argparse.ArgumentParser(sys.argv[0])

argparser.add_argument('--train_file', help='Trained file', default="data/smsspamcollection/train.csv", type=str)

argparser.add_argument('--dev_file', help='Developed file', default="data/smsspamcollection/test.csv", type=str)

argparser.add_argument('--test_file', help='Tested file', default="data/smsspamcollection/test.csv", type=str)

argparser.add_argument("--tfidf", action='store_true', default=False, help="tfidf flag")

argparser.add_argument("--use_hash", action='store_true', default=False, help="hashing flag")

argparser.add_argument("--scaler", action='store_true', default=False, help="scale flag")

argparser.add_argument('--ml_cls', help='Machine learning classifier', default="MLP", type=str)

argparser.add_argument('--model_dir', help='Model dir', default="data/smsspamcollection/", type=str)

args, unknown = argparser.parse_known_args()

model_dir, _ = os.path.split(args.model_dir)

if not os.path.exists(args.model_dir):
    os.mkdir(args.model_dir)
args.model_name = os.path.join(args.model_dir, args.ml_cls + ".pickle")

app = Flask(__name__)

app.secret_key = 'COMP4312_FinalProject'

model_api = load(args.model_name)


def sigterm_handler(_signo, _stack_frame):
    print(str(datetime.datetime.now()) + ': Received SIGTERM')


def sigint_handler(_signo, _stack_frame):
    print(str(datetime.datetime.now()) + ': Received SIGINT')
    sys.exit(0)


signal.signal(signal.SIGTERM, sigterm_handler)
signal.signal(signal.SIGINT, sigint_handler)


@app.route('/')
def index():
    return render_template("index.html")



@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/database')
def database():
    # When deployed to App Engine, the `GAE_ENV` environment variable will be
    # set to `standard`
    if os.environ.get('GAE_ENV') == 'standard':
        # If deployed, use the local socket interface for accessing Cloud SQL
        unix_socket = '/cloudsql/{}'.format(db_connection_name)
        cnx = pymysql.connect(user=db_user, password=db_password,
                              unix_socket=unix_socket, db=db_name)
    else:
        # If running locally, use the TCP connections instead
        # Set up Cloud SQL Proxy (cloud.google.com/sql/docs/mysql/sql-proxy)
        # so that your application can use 127.0.0.1:3306 to connect to your
        # Cloud SQL instance
        host = '127.0.0.1'
        cnx = pymysql.connect(user=db_user, password=db_password,
                              host=host, db=db_name)

    with cnx.cursor() as cursor:
        cursor.execute('select demo_txt from demo_tbl;')
        result = cursor.fetchall()
        current_msg = result[0][0]
    cnx.close()

    return str(current_msg)
    # [END gae_python37_cloudsql_mysql]
    return render_template("database.html")

@app.route('/inference', methods=('GET', 'POST'))
def inference():
    if request.method == 'POST':
        text = request.form['input']
        # content = request.form['content']

        if not text:
            flash('text is required!')
        else:
            label = model_api.predict([text]).tolist()[0]
            prob = model_api.predict_proba([text]).max()
            result = dict()
            result["input"] = text
            result["output"] = label
            result["probability"] = prob
            app.logger.info("model_output: " + str(result))
            return render_template('inference.html', label=label, prob=prob)
    return render_template('inference.html', label="NA", prob="NA")


if __name__ == '__main__':
    app.run()
