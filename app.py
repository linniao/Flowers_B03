from flask import Flask, request, render_template,jsonify
from datetime import timedelta
from connector1 import interface as inter1
from connector2 import interface as inter2
from connector3 import interface as inter3
import numpy as np
import os
from PIL import Image
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
    b = Image.open(a)
    b = np.array(b)
    # print(b)
    b = b.repeat(8, axis=0).repeat(8, axis=1)
    b = Image.fromarray(b.astype('uint8')).convert('RGB')
    b.save(os.path.join('./static/result', "orign.jpg"))

    return jsonify({'filename1':  "./static/result/"+"orign.jpg",
                    'filename2': inter1(a),
                    'filename3': inter3(a)
                    })


if __name__ == '__main__':
    app.run()
