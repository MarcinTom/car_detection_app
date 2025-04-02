# Car Detection Application

This is a Flask application that allows users to upload images and count the number of cars in the image using the Mask R-CNN ResNet50 model from PyTorch.

## Prerequisites

Before starting, ensure you have the following installed:
- Python 3.13 or later
- `pip` (Python package manager)

## Installation Instructions

1. **Clone the repository or download the project code:**

2. **Create a virtual environment:**
-  For Windows:
    ```powershell
    python -m venv venv
    ```

- For Linux/macOS:
    ```bash
    python3 -m venv venv
    ```

3. **Activate the virtual environment:**
- On **Windows**:
  ```
  venv\Scripts\activate
  ```
- On **Linux/MacOS**:
  ```
  source venv/bin/activate
  ```

4. **Install required packages:**
Use the `requirements.txt` file to install all necessary dependencies with
```
pip install -r requirements.txt
```

## Running the Application on Windows

1. Ensure the virtual environment is active (see step 3 above).
2. Start the Flask application through Terminal: 
```
python app.py
```
3. Open your web browser and for local use navigate to: 
```
http://127.0.0.1:5000/
```

The below code in app.py allows also the access to the app through the Internet.
Target address will be  shown in Terminal and depends on your network settings.
If you don't want it remove host and port setup

```
port = int(os.environ.get('PORT', 5000))
app.run(debug=True, host="0.0.0.0", port=port)
```
4. Upload a picture with cars
5. Verify results

## Running the Application on Docker

1. Verify Dockerfile setup
2. Execute create_docker.sh in Terminal or run below
```
docker build -t car_app .
```
3. Run Docker image allocating prots in container and on the local machine: 
```
docker run -p 5000:5000 -d flask_docker
```

4. Visit this website to use the app

```
localhost:5000
```
5. Upload a picture with cars
6. Verify results