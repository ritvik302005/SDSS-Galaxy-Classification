
# SDSS Galaxy Classification

## Project Overview
This project implements a machine learning based system to classify galaxies using features from the SDSS dataset. A Flask web application is used to provide an interactive interface for prediction.

## Technologies Used
- Python
- Flask
- HTML, CSS
- Scikit-learn
- Pandas, NumPy

## Project Structure
- app.py : Flask backend
- templates/ : HTML files
- static/ : CSS and assets
- model files : trained ML model and scaler

## How to Run
1. Install dependencies using `pip install -r requirements.txt`
2. Run the application using `python app.py`
3. Open browser and go to `http://127.0.0.1:5000`

## Output
The system takes galaxy features as input and predicts the galaxy class.
