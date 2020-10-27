from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("../COMP4312-AnimalRecognition/templates/index.html")


@app.route('/about')
def about():
    return render_template("../COMP4312-AnimalRecognition/templates/about.html")


@app.route('/inference', methods=('GET','POST'))
def inference():
    return render_template("../COMP4312-AnimalRecognition/templates/inference.html")


if __name__ == '__main__':
    app.run()
