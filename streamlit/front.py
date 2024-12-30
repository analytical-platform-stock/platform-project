import streamlit as st
import requests
import logging
from logging.handlers import RotatingFileHandler
import os
import pandas as pd
import nbformat
from nbconvert import HTMLExporter
from typing import Dict
import io
from PIL import Image

# Настройка логирования
LOG_FILE = "logs/frontend.log"
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("StreamlitApp")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)  # 5MB max per log file
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Константы
API_URL = "http://127.0.0.1:8000"

NOTEBOOK_PATH = "eda.ipynb"


def get_models():
    """Получить список всех моделей"""
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            return response.json()
        st.error(f"Ошибка загрузки моделей: {response.text}")
    except Exception as e:
        logger.error(f"Ошибка получения моделей: {e}")
        st.error("Ошибка связи с сервером")


def set_active_model(model_id):
    """Установить активную модель"""
    try:
        response = requests.post(f"{API_URL}/set", params={"model_id": model_id})
        if response.status_code == 200:
            st.success(f"Модель {model_id} установлена как активная.")
        else:
            st.error(f"Ошибка установки модели: {response.text}")
    except Exception as e:
        logger.error(f"Ошибка установки активной модели: {e}")
        st.error("Ошибка связи с сервером")


def fit_model(hyperparameters):
    """Запустить обучение модели"""
    try:
        response = requests.post(f"{API_URL}/fit", params=hyperparameters["params"], files=hyperparameters["files"])
        if response.status_code == 202:
            st.success("Обучение модели началось.")
        else:
            st.error(f"Ошибка запуска обучения: {response.text}")
    except Exception as e:
        logger.error(f"Ошибка обучения модели: {e}")
        st.error("Ошибка связи с сервером")


def plot_scores(params):
    """Вывод кривых обучения"""
    try:
        response = requests.get(f"{API_URL}/scores", params=params)
        if response.status_code == 200:
            # Открываем изображение из байтового потока
            image = Image.open(io.BytesIO(response.content))

            # Отображаем изображение на странице
            st.image(image, caption="Загруженный график", use_container_width=True)
        else:
            st.error(f"Ошибка при загрузке графика: {response.json().get('error', 'Неизвестная ошибка')}")
    except Exception as e:
        st.error(f"Произошла ошибка: {e}")


def predict(files):
    """Выполнить прогнозирование"""
    try:
        response = requests.post(f"{API_URL}/predict", files=files)
        if response.status_code == 200:
            return response.json()
        st.error(f"Ошибка выполнения предсказания: {response.text}")
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        st.error("Ошибка связи с сервером")


def display_notebook(notebook_path):
    """Функция для отображения содержимого Jupyter Notebook в Streamlit"""
    try:
        # Чтение .ipynb файла
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_content = nbformat.read(f, as_version=4)

        # Конвертация в HTML
        html_exporter = HTMLExporter()
        (body, resources) = html_exporter.from_notebook_node(notebook_content)

        return body
    except Exception as e:
        st.error(f"Ошибка при обработке .ipynb файла: {e}")
        return None


def fetch_model_info():
    """Получить информацию о модели через API"""
    try:
        response = requests.get(f"{API_URL}/model_info")  # Создайте такую ручку в FastAPI
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка получения информации о модели: {response.text}")
            return {}
    except Exception as e:
        logger.error(f"Ошибка получения информации о модели: {e}")
        st.error("Ошибка связи с сервером")
        return {}

def fetch_training_curves():
    """Получить данные для кривых обучения"""
    try:
        response = requests.get(f"{API_URL}/training_curves")  # Создайте такую ручку в FastAPI
        if response.status_code == 200:
            return response.json()  # Должно возвращать JSON с метриками
        else:
            st.error(f"Ошибка получения кривых обучения: {response.text}")
            return {}
    except Exception as e:
        logger.error(f"Ошибка получения кривых обучения: {e}")
        st.error("Ошибка связи с сервером")
        return {}


# Пользовательский интерфейс Streamlit
st.title("Machine Learning Service Dashboard")

st.sidebar.title("Навигация")
page = st.sidebar.radio("Выберите страницу", ["Dataset", "EDA", "Model Management", "Inference"])

if page == "Dataset":
    st.header("Загрузка датасета")
    uploaded_file = st.file_uploader("Загрузите файл датасета (CSV)", type=["csv"])
    if uploaded_file:
        try:
            DATA_TRAIN = pd.read_csv(uploaded_file, sep=";")
            st.dataframe(DATA_TRAIN)
            st.success("Датасет успешно загружен")
            logger.info("Dataset загружен.")
            st.session_state.DATA_TRAIN = DATA_TRAIN
        except Exception as e:
            st.error("Ошибка загрузки датасета.")
            logger.error(f"Ошибка загрузки датасета: {e}")

elif page == "EDA":
    st.header("Exploratory Data Analysis (EDA)")

    st.write("Посмотрите предварительно выполненную Jupyter Notebook-аналитику:")
    notebook_html = display_notebook(NOTEBOOK_PATH)
    
    if notebook_html:
        # Используем HTML компонент Streamlit для отображения содержимого
        st.components.v1.html(notebook_html, height=800, scrolling=True)
    else:
        st.error("Не удалось загрузить Jupyter Notebook для отображения.")

elif page == "Model Management":
    hyperparameters: Dict = {}

    st.header("Управление моделями и их обучение")

    st.subheader("Текущие модели")
    models = get_models()
    models_ids = [model["id"] for model in models]
    if models:
        st.write(models)
        selected_model = st.selectbox("Выберите модель для установки", models_ids)
        if st.button("Установить активную модель"):
            set_active_model(selected_model)
    else:
        st.write("Нет моделей для обучения")

    st.subheader("Обучение модели")
    model_types = ["LinearRegression", "ARIMA"]
    select_type = st.selectbox("Выберите тип обучаемой модели", model_types)
    if select_type == "LinearRegression":
        selected_model = st.selectbox("Выберите модель для обучения", models_ids)
        hyperparameters["model_id"] = selected_model
    else:
        st.write("Введите гиперпараметры для обучения модели.")
        p = st.slider("p (порядок авторегрессии)", 1, 10, value=2)
        d = st.slider("d (порядок дифференцирования)", 1, 10, value=2)
        q = st.slider("q (порядок скользящего среднего)", 1, 10, value=2)
        hyperparameters["p"] = p
        hyperparameters["d"] = d
        hyperparameters["q"] = q

    if select_type == "LinearRegression":
        if st.button("Запустить обучение"):
            csv_buffer = io.StringIO()
            st.session_state.DATA_TRAIN.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            fit_model({"params": hyperparameters, "files": {"file": ('temp_data.csv', csv_buffer, 'text/csv')}})
            plot_scores({"model_id": hyperparameters["model_id"]})
    else:
        st.write("На текущий момент обучение моделей данного типа не доступно")

elif page == "Inference":
    st.header("Инференс (Предсказание)")
    st.write("Задайте данные для предсказания.")

    uploaded_file = st.file_uploader("Загрузите файл датасета (CSV)", type=["csv"])
    if uploaded_file:
        try:
            DATA_PREDICT = pd.read_csv(uploaded_file, sep=";")
            st.success("Датасет успешно загружен")
            logger.info("Dataset загружен.")
            st.session_state.DATA_PREDICT = DATA_PREDICT
        except Exception as e:
            st.error("Ошибка загрузки датасета.")
            logger.error(f"Ошибка загрузки датасета: {e}")
    if st.button("Выполнить предсказание"):
        try:
            csv_buffer = io.StringIO()
            st.session_state.DATA_PREDICT.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            result = predict({"file": ('temp_data.csv', csv_buffer, 'text/csv')})
            if result:
                st.write("Результат предсказания:")
                st.json(result)
        except Exception as e:
            st.error("Ошибка обработки данных для предсказания.")
            logger.error(f"Ошибка обработки данных для предсказания: {e}")