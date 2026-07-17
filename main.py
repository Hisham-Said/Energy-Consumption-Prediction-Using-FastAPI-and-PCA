from fastapi import FastAPI, Request, Form
import numpy as np
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib

app = FastAPI(title="Energy Consumption Prediction")

# Load saved pipeline
saved_objects = joblib.load("model.joblib")
model = saved_objects["model"]
pca = saved_objects["pca"]
scaler = saved_objects["scaler"]

# Templates folder
templates = Jinja2Templates(directory="templates")

# Static folder
app.mount("/static", StaticFiles(directory="static"), name="static")


# -------------------------
# Home Page
# -------------------------
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": "Energy Consumption Prediction Dashboard"
        }
    )
@app.get("/dashboard")
async def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "title": "Dashboard"
        }
    )
@app.get("/predict")
async def predict_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
            "prediction": None
        }
    )


@app.post("/predict")
async def predict(
    request: Request,
    Lagging_Current_Reactive_Power_kVarh: float = Form(...),
    Leading_Current_Reactive_Power_kVarh: float = Form(...),
    CO2_tCO2: float = Form(...),
    Lagging_Current_Power_Factor: float = Form(...),
    Leading_Current_Power_Factor: float = Form(...),
    NSM: float = Form(...),
    WeekStatus: int = Form(...),
    Day_of_week: int = Form(...),
    Load_Type: int = Form(...)
):

    data = np.array([[
        Lagging_Current_Reactive_Power_kVarh,
        Leading_Current_Reactive_Power_kVarh,
        CO2_tCO2,
        Lagging_Current_Power_Factor,
        Leading_Current_Power_Factor,
        NSM,
        WeekStatus,
        Day_of_week,
        Load_Type
    ]])

    # Apply preprocessing exactly like training
    data_scaled = scaler.transform(data)
    data_pca = pca.transform(data_scaled)

    prediction = model.predict(data_pca)[0]

    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
            "prediction": round(float(prediction), 2)
        }
    )