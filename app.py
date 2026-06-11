import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle

app = FastAPI()
model = pickle.load(open('model.pkl', 'rb'))
templates = Jinja2Templates(directory="templates")

@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"prediction_text": ""})

@app.post('/predict', response_class=HTMLResponse)
def predict(
    request: Request,
    mean_radius: float = Form(...),
    mean_texture: float = Form(...),
    mean_perimeter: float = Form(...),
    mean_area: float = Form(...),
    mean_smoothness: float = Form(...),
    mean_compactness: float = Form(...),
    mean_concavity: float = Form(...),
    mean_concave_points: float = Form(...),
    mean_symmetry: float = Form(...),
    mean_fractal_dimension: float = Form(...),
    radius_error: float = Form(...),
    texture_error: float = Form(...),
    perimeter_error: float = Form(...),
    area_error: float = Form(...),
    smoothness_error: float = Form(...),
    compactness_error: float = Form(...),
    concavity_error: float = Form(...),
    concave_points_error: float = Form(...),
    symmetry_error: float = Form(...),
    fractal_dimension_error: float = Form(...),
    worst_radius: float = Form(...),
    worst_texture: float = Form(...),
    worst_perimeter: float = Form(...),
    worst_area: float = Form(...),
    worst_smoothness: float = Form(...),
    worst_compactness: float = Form(...),
    worst_concavity: float = Form(...),
    worst_concave_points: float = Form(...),
    worst_symmetry: float = Form(...),
    worst_fractal_dimension: float = Form(...)
):
    input_features = [
        mean_radius, mean_texture, mean_perimeter, mean_area,
        mean_smoothness, mean_compactness, mean_concavity,
        mean_concave_points, mean_symmetry, mean_fractal_dimension,
        radius_error, texture_error, perimeter_error, area_error,
        smoothness_error, compactness_error, concavity_error,
        concave_points_error, symmetry_error, fractal_dimension_error,
        worst_radius, worst_texture, worst_perimeter, worst_area,
        worst_smoothness, worst_compactness, worst_concavity,
        worst_concave_points, worst_symmetry, worst_fractal_dimension
    ]
    features_value = [np.array(input_features)]

    features_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']

    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)

    if output == 0:
        res_val = "** breast cancer **"
    else:
        res_val = "no breast cancer"

    return templates.TemplateResponse(request, "index.html", {
        "prediction_text": "Patient has {}".format(res_val)
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
