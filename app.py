from flask import Flask, request, render_template, flash, redirect
from werkzeug.utils import secure_filename
import os

from model.model import predict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}


@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
 
    file = request.files['file']
    
    if file.filename == '':
        return render_template('error.html', message="No file selected")
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        car_count = predict(file_path)
        
        return render_template('result.html', car_count=car_count)
    else:
        return render_template('error.html', message="Wrong file type")


def allowed_file(filename):
    return filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
