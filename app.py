from flask import Flask, request, render_template, flash, redirect
from werkzeug.utils import secure_filename
import os

from model.model import predict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}


@app.route('/')
def index():
    """
    Render the homepage.

    This route renders the `upload.html` template, which serves as the homepage 
    where users can upload an image file.

    Returns:
        str: The rendered HTML content of the `upload.html` template.
    """
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and vehicle prediction.

    This route processes an uploaded image file, checks its validity, saves it to 
    the server, and uses a pre-trained model to predict the number of vehicles 
    in the image. The results are displayed on the `result.html` template.

    If the file is invalid or not selected, it redirects to an error page.

    Args:
        None (data is retrieved from `request.files`).

    Returns:
        str: The rendered HTML content of either:
            - `result.html` with the vehicle count if successful.
            - `error.html` with an error message if unsuccessful.
    
    Notes:
        - Only files with extensions defined in `ALLOWED_EXTENSIONS` are accepted.
        - The uploaded file is saved securely using `secure_filename`.
    """
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
    """
    Check if a file's extension is allowed.

    This helper function verifies whether the given filename has an extension 
    that is included in the application's allowed extensions.

    Args:
        filename (str): The name of the file to check.

    Returns:
        bool: True if the file extension is allowed, False otherwise.
    """
    return filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
