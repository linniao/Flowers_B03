from flask import Flask, request, render_template,jsonify
from datetime import timedelta
from connector1 import interface


import os
from werkzeug.utils import secure_filename

app = Flask(__name__)


app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


@app.route('/')
def login():
    return render_template('login.html')


@app.route('/success/<name>')
def success(name):
    return "Welcome  %s" % name


@app.route('/receive', methods=['GET', 'POST'])
def receive():
    if not os.path.exists('./static/result'):
        os.makedirs('./static/result')
    a = request.files['file']

    filename = secure_filename(a.filename)
    a.save(os.path.join('./static/result', filename))
    print("b")
    # return jsonify({'filename1': interface(a),
    #                 'filename2': "./static/result/"+filename
    #                 })
    return jsonify({'filename1': interface(a),
                    'filename2': "./static/result/"+filename
                    })

if __name__ == '__main__':
    app.run()
