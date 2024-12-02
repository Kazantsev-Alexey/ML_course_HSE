import pandas as pd
import io
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import pickle

app = FastAPI()

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('impute_values.pkl', 'rb') as f:
    impute_values = pickle.load(f)

class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

@app.get("/")
def read_root():
    return {"message": "Эндпоинт /predict_item - предсказание цены одного автомобиля. Эндпоинт /predict_items - предсказание цены нескольких автомобилей."}
    
def preprocess_data(data):
    data = data.drop(['name'],axis=1)
    
    # Мы заполняли пустоты с помощью медианы, для предикта будем использовать те же значения
    for col, value in impute_values.items():
        data[col] = data[col].fillna(value)
    
    # все преобразования по аналогии с EDA
    data.mileage = data.mileage.apply(lambda x: str(x).replace(' kmpl', '').replace('km/kg','')).astype(float)
    data.engine = data.engine.apply(lambda x: str(x).replace(' CC', '')).astype(float)
    data.max_power = data.max_power.apply(lambda x: str(x).replace(' bhp', '')).astype(float)
    data.seats = data.seats.fillna(data.seats.median()).astype(int)
    data.engine = data.engine.fillna(data.seats.median()).astype(int)
    data_values_only = data.select_dtypes(include=['float','int'])
    data_values_only = pd.DataFrame(scaler.transform(data_values_only))
    data_values_only.columns = data.select_dtypes(include=['float','int']).columns

    cats = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
    data_catagories = encoder.transform(data[cats])
    encoded_data = pd.DataFrame(data_catagories, columns=encoder.get_feature_names_out(cats), index=data.index)

    result_df = pd.concat([data_values_only,encoded_data],axis=1)
    return result_df

@app.post("/predict_item")
def predict_item(item: Item):
    data = pd.DataFrame([item.dict()])
    data = preprocess_data(data)
    prediction = model.predict(data)
    return prediction[0]

@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)):
    try:
        data = pd.read_csv(file.file, index_col=0)
        data = preprocess_data(data)
        predictions = model.predict(data)
        data['selling_price'] = predictions
        output = io.StringIO()
        data.to_csv(output, index=False)
        output.seek(0)
        return StreamingResponse(
            output,
            media_type='text/csv',
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})