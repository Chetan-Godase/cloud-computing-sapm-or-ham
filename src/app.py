import os
import argparse
from flask import Flask, request, render_template, flash
from baselines import load
import signal
import datetime
import sys

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
    app.run(host='0.0.0.0')
