from flask import Flask, request, render_template
from prediction import CIFAR10
from matplotlib import image
import os

app = Flask(__name__)
UPLOAD_FOLDER = './saved_images'
# ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

c = CIFAR10()


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/prd', methods=['GET', 'POST'])
def predict():
    img = request.form['image']
    img = image.imread(os.path.join(os.getcwd(),'static','test_images',img))
    category = c.predict_category(img)
    return render_template('index.html', category=category)


if __name__ == '__main__':
    app.run(debug=True)
