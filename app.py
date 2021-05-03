from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/login')
def login():
    return render_template('doclogin.html')
@app.route('/info')
def info():
    return render_template('1.html')

@app.route('/input')
def input():
    return render_template('2.html')

if __name__ == '__main__':
    app.run()