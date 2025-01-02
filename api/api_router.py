import os
import logging
import pickle
import shutil
from sklearn.linear_model import LinearRegression
from fastapi import APIRouter, FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from typing import List
from pydantic import BaseModel
from contextlib import asynccontextmanager
from multiprocessing import Process
import pandas as pd
import math
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
import io
import time

router = APIRouter()


# Установка логирования
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Переменные для моделей
models = {}
active_model_id = None

class ModelInfo(BaseModel):
    id: str
    name: str
    type: str
    status: str

class PredictionRequest(BaseModel):
    data: List[float]

class PredictionResponse(BaseModel):
    predictions: List[float]

# Загрузка моделей из папки models/
MODEL_DIRECTORY = "api/models"
os.makedirs(MODEL_DIRECTORY, exist_ok=True)

def load_models():
    global models
    for company in ["GAZP", "LKOH", "NVTK", "ROSN", "SNGS", "TATN"]:
        for model_type in ["LinReg"]: # Планируется добавление других типов моделей
            model_path = os.path.join(MODEL_DIRECTORY, f"{model_type}_{company}.pkl")
            if os.path.exists(model_path):
                with open(model_path, "rb") as file:
                    models[f"{model_type}_{company}"] = {
                        "model": pickle.load(file),
                        "type": model_type,
                        "company": company
                    }
            else:
                print(os.getcwd())

def get_model_info():
    return [
        ModelInfo(
            id=model_id,
            name=f"Model for {model_data['company']} ({model_data['type']})",
            type=model_data['type'],
            status="loaded"
        )
        for model_id, model_data in models.items()
    ]

# Инициализация с Lifespan Events
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Инициализация моделей...")
    load_models()
    if models:
        global active_model_id
        active_model_id = list(models.keys())[0]  # Устанавливаем первую модель как активную
    logger.info(f"Доступные модели: {list(models.keys())}")
    yield
    logger.info("Остановка приложения и выгрузка моделей.")


# ручки API
@router.get("/models", response_model=List[ModelInfo])
def get_models():
    """Список текущих моделей"""
    return get_model_info()

@router.post("/set", response_model=ModelInfo)
def set_active_model(model_id: str):
    """Установка активной модели"""
    global active_model_id
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Модель не найдена")
    active_model_id = model_id
    model_data = models[model_id]
    return ModelInfo(
        id=model_id,
        name=f"Model for {model_data['company']} ({model_data['type']})",
        type=model_data['type'],
        status="active"
    )

@router.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...)):
    """Предикт активной модели"""
    if not active_model_id:
        raise HTTPException(status_code=500, detail="Активная модель не установлена")
    model = models[active_model_id]["model"]
    try:
        df = pd.read_csv(file.file)

        predictions = model.predict(df.values)  # Преобразуем данные в формат массива
        return PredictionResponse(predictions=predictions.tolist())
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        raise HTTPException(status_code=500, detail="Ошибка предсказания")


def plot_scores(model, model_id, X_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, 
                                                         train_sizes=np.linspace(0.1, 1.0, 10),
                                                         scoring='neg_mean_squared_error',
                                                         cv=5)

    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    test_scores_std = test_scores.std(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color='g')

    plt.title('Learning Curve for Linear Regression')
    plt.xlabel('Training Size')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend(loc='best')
    plt.grid()

    model_path = os.path.join(MODEL_DIRECTORY, f"{model_id}_learning_curve.png")
    plt.savefig(model_path, format='png', dpi=300)  # Указываем разрешение в dpi
    
    plt.close()  


def train_linear_regression(file_path: str, model_id: str, test_size=0.15, lag_start=1, lag_end=2):
    """Процесс обучения модели"""
    logger.info(f"Начало обучения модели {model_id}")
    try:
        # Загружаем данные из CSV
        data = pd.read_csv(file_path)
        #data['time'] = pd.to_datetime(data['time'])
        #data.set_index('time', inplace=True)
        test_data_size = int(len(data) * test_size)

        # Тренировочная часть — все строки, кроме последних 15%
        train_data = data.iloc[:-test_data_size]
        # Тестовая часть — последние 15% строк
        test_data = data.iloc[-test_data_size:]

        # добавляем лаги исходного ряда в качестве признаков
        for i in range(lag_start, lag_end):
            data[f"lag_{i}"] = data['close'].shift(i)
        data = data.dropna()


        # разбиваем весь датасет на тренировочную и тестовую выборку
        X_train = data.head(math.ceil(int(len(data) * 0.85))).drop(["close"], axis=1)
        y_train = data.head(math.ceil(int(len(data) * 0.85)))["close"]
        X_test = data.tail(int(len(data) * 0.15)).drop(["close"], axis=1)
        y_test = data.tail(int(len(data) * 0.15))["close"]

        # Создаем и обучаем модель линейной регрессии
        model = LinearRegression()
        model.fit(X_train, y_train)

        #Создаем и сохраняем графики обучения модели линейной регрессии
        plot_scores(model, model_id, X_train, y_train)

        # Сохраняем обученную модель
        model_path = os.path.join(MODEL_DIRECTORY, f"{model_id}.pkl")
        with open(model_path, "wb") as file:
            pickle.dump(model, file)


        models[model_id] = {"model": model, "type": "linear_regression", "company": model_id.split('_')[0]}

        logger.info(f"Обучение модели {model_id} завершено")
    except Exception as e:
        logger.error(f"Ошибка обучения модели {model_id}: {e}")

@router.post("/fit")
def fit(
    model_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Запуск обучения модели"""
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Модель не найдена")


    temp_file_path = os.path.join(MODEL_DIRECTORY, f"temp_{model_id}.csv")
    with open(temp_file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Запускаем процесс обучения
    process = Process(target=train_linear_regression, args=(temp_file_path, model_id))
    process.start()
    background_tasks.add_task(process.join)
    
    return JSONResponse(content={"status": "Запущено обучение"}, status_code=202)

@router.get("/scores")
def get_scores(model_id: str):
    pkl_file_path = os.path.join(MODEL_DIRECTORY, f"{model_id}_learning_curve.png")

    if not os.path.isfile(pkl_file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=pkl_file_path, media_type='image/png', filename=f"{model_id}_learning_curve.png")