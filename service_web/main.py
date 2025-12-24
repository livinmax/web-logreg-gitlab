import os
import joblib
import pandas as pd
import numpy as np
from typing import List
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.base import BaseEstimator, TransformerMixin
import uvicorn
import __main__

class DateCyclicalTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        dt_col = pd.to_datetime(X.iloc[:, 0], errors='coerce').fillna(pd.Timestamp('2000-01-01'))
        df_out = pd.DataFrame(index=X.index)
        df_out['year'] = dt_col.dt.year
        df_out['is_weekend'] = (dt_col.dt.dayofweek >= 5).astype(int)
        month, dow = dt_col.dt.month, dt_col.dt.dayofweek
        df_out['month_sin'] = np.sin(2 * np.pi * month / 12)
        df_out['month_cos'] = np.cos(2 * np.pi * month / 12)
        df_out['day_sin'] = np.sin(2 * np.pi * dow / 7)
        df_out['day_cos'] = np.cos(2 * np.pi * dow / 7)
        return df_out
    def get_feature_names_out(self, input_features=None):
        return np.array(['year', 'is_weekend', 'month_sin', 'month_cos', 'day_sin', 'day_cos'])

class VIFSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=10.0, skip_pattern='flag_'):
        self.threshold, self.skip_pattern, self.columns_to_keep_ = threshold, skip_pattern, None
    def fit(self, X, y=None): return self
    def transform(self, X):
        curr_X = pd.DataFrame(X)
        curr_X.columns = curr_X.columns.astype(str)
        if hasattr(self, 'columns_to_keep_') and self.columns_to_keep_ is not None:
            return curr_X[self.columns_to_keep_]
        return curr_X

__main__.DateCyclicalTransformer = DateCyclicalTransformer
__main__.VIFSelector = VIFSelector

app = FastAPI()
templates = Jinja2Templates(directory="templates")

POSSIBLE_PATHS = [
    "/app/service_model/full_model_pipeline.joblib",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'service_model', 'full_model_pipeline.joblib'),
    "../service_model/full_model_pipeline.joblib"]

model = None
for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            print(f"--- SUCCESS: Model loaded from {path}")
            break
        except Exception as e:
            print(f"--- ERROR loading from {path}: {e}")

if model is None:
    print("--- CRITICAL ERROR: Model could not be loaded!")

VENUES = ["ALL", "HC0", "HC1", "HC2", "HC3", "HC4", "HC5", "HC6", "HC7", "HC8", "HC9", "HC10", "PP0", "PP1", "KA0",          
          "KA1", "KA2", "KA3", "OF0", "OF1", "OF2", "OF3", "OF4", "OF5", "OF6", "OF7", "OS0", "OS1", "OT0"]
MEALS = [f"ME{i}" for i in range(11)]
MANAGERS = [f"MA{i}" for i in range(16)]

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request, "venues": VENUES, "meals": MEALS, "managers": MANAGERS, "prediction": None})

@app.post("/predict")
async def handle_form(
        request: Request,
        status: str = Form(...),
        date_received: str = Form(...),
        source_method: str = Form(...),
        is_agency: bool = Form(False),
        event_format: str = Form(...),
        date_start: str = Form(...),
        date_end: str = Form(...),
        guest_count: int = Form(...),
        is_venue_closed: bool = Form(False),
        is_restaurant_closed: bool = Form(False),
        budget: float = Form(...),
        venues: List[str] = Form(default=[]),
        meals: List[str] = Form(default=[]),
        managers: List[str] = Form(default=[])):

    try:
        d_rec = pd.to_datetime(date_received)
        d_start = pd.to_datetime(date_start)
        d_end = pd.to_datetime(date_end)

        data = {
            "date_request": d_rec, "group_source": source_method,
            "flag_event_agency": 1 if is_agency else 0, "event_category": event_format,
            "date_event_start": d_start, "date_event_end": d_end, "cnt_guest": guest_count,
            "flag_place_closure": 1 if is_venue_closed else 0,
            "flag_restaurant_closure": 1 if is_restaurant_closed else 0,
            "event_budget": budget}

        for v in VENUES: data[v] = 1 if v in venues else 0
        for m in MEALS: data[m] = 1 if m in meals else 0
        for mgr in MANAGERS: data[mgr] = 1 if mgr in managers else 0

        df = pd.DataFrame([data])
        df['duration_event'] = (d_end - d_start).days
        df['duration_process'] = (d_end - d_rec).days
        df['flag_event_budget'] = 1 if budget != 0 else 0
        df['cnt_place'] = len(venues)
        df['cnt_meals'] = len(meals)
        df['cnt_manager'] = len(managers)

        cols_to_fix = df.select_dtypes(include=['int64', 'int32']).columns
        df[cols_to_fix] = df[cols_to_fix].astype(float)

        prediction_text = "Model is not loaded"
        prob = 0
        
        if model:
            prob_val = model.predict_proba(df)[0][1]
            prob = round(prob_val * 100, 2)
            prediction_text = f"Probability for status 'Released': {prob}%"

    except Exception as e:
        prediction_text = f"Processing Error: {str(e)}"
        prob = 0

    return templates.TemplateResponse("index.html", {
        "request": request, "venues": VENUES, "meals": MEALS, "managers": MANAGERS,
        "prediction": prediction_text, "probability": prob})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)