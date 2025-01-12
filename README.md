<p align="center">
      <img src=https://media.licdn.com/dms/image/v2/C4D12AQGXZt4AmC6F4w/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1652530267151?e=2147483647&v=beta&t=Td0hN85YQ2bsoVGjmEWZThIJhdZV5ik23xxOKggBssY width="726">
</p>

<p align="center">
   <img src=https://img.shields.io/badge/Specialization-Russian_blue_chips-purple alt="Specialization">
   <img src=https://img.shields.io/badge/Deeplearning-2_models-blue alt="Models">
   <img src=https://img.shields.io/badge/License-MIT-green alt="License">
</p>

# Machine Learning Service

## 📋 Краткое описание

Проект будет состоять из двух направлений: предсказание временных рядов стоимости ценных бумаг и анализ новостного фона компаний. Платформа ориентирована на российский рынок. В конечном итоге планируется платформа, на основе которой пользователь может принять решение о приобретении акций компании. Этот проект включает API на FastAPI и визуальный интерфейс на Streamlit для управления моделями и анализа данных.

## 🎯 Функционал

### API
- Получение списка моделей.
- Установка активной модели.
- Выполнение предсказаний.
- Обучение новой модели.

![API Demo](docs/api.gif)

### Streamlit
- Загрузка датасета.
- Демонстрация аналитики (EDA).
- Управление моделями.
- Инференс данных.

![Streamlit Demo](docs/streamlit.gif)

## Types of problems

1. Classification (предсказание меток классов новостных заголовков)
2. Regression (предсказание стоимости акции)

## Models

1. Bert-base-multilingual-uncased-sentiment
2. ARIMA

## Shares

1. Роснефть
2. Газпром
3. НОВАТЭК
4. Лукоил
5. Сургутнефтегаз
6. Татнефть

## 🚀 Инструкция по запуску

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/your-repo/project.git
   cd project
   ```

2. Настройте окружение:
   - Создайте файл `.env` и заполните его параметрами, включая API-токены, ключи и другие переменные, необходимые для работы приложения. Пример содержимого `.env`:
     ```env
     API_URL=http://localhost:8001
     DB_URL=postgresql://user:password@localhost/db
     SECRET_KEY=your_secret_key
     ```

3. Убедитесь, что у вас установлены все необходимые зависимости для Docker и Docker-Compose.

4. Соберите и запустите контейнеры:
   ```bash
   docker-compose up --build
   ```
   ![Docker-compose diagram](docs/docker_compose_diagram.png)
   ![Docker Example](docs/docker.gif)

5. Проверьте, что все сервисы работают корректно. Для этого можно зайти на:
   - FastAPI: [http://localhost:8001](http://localhost:8001)
   - Streamlit: [http://localhost:8501](http://localhost:8501)

6. Запустите тестовые запросы через Postman или командную строку, чтобы убедиться в корректной работе API.

## 📊 Примеры работы

### Работа API

#### Получение списка моделей
Выполните следующий запрос, чтобы получить список всех доступных моделей:
```bash
curl -X GET "http://localhost:8001/models"
```
**Пример ответа:**
```json
[
  {
    "id": "LinReg_GAZP",
    "name": "Model for GAZP (LinReg)",
    "type": "LinReg",
    "status": "loaded"
  }
]
```

#### Выполнение предсказания
Предоставьте CSV-файл с данными для анализа и выполните запрос:
```bash
curl -X POST "http://localhost:8001/predict" -F "file=@data.csv"
```
**Пример ответа:**
```json
{
  "predictions": [123.45, 678.90]
}
```
![API Example](docs/api.gif)

### Работа Streamlit

#### Загрузка датасета и выполнение анализа
1. Перейдите на [http://localhost:8501](http://localhost:8501).
2. На странице "Dataset" загрузите CSV-файл с данными.
3. Проверьте загруженные данные, которые отобразятся в таблице.

#### Демонстрация EDA
1. Перейдите на страницу "EDA".
2. Ознакомьтесь с аналитикой и визуализациями, представленными в интерфейсе.

#### Выполнение предсказания
1. Перейдите на страницу "Inference".
2. Загрузите CSV-файл с данными.
3. Нажмите кнопку "Выполнить предсказание" и ознакомьтесь с результатами, которые появятся в формате JSON.

![Streamlit Example](docs/streamlit.gif)

## Developers

**Участники**:
- Золотухин Дмитрий — [@tobdin](https://t.me/tobdin), [dmitriygeo](https://github.com/dmitriygeo)
- Иваненков Лев — [@ivanenkovLev](https://t.me/ivanenkovLev), [Shredian](https://github.com/Shredian)
- Карпов Роман — [@KarpovRomanQWE](https://t.me/KarpovRomanQWE), [KRVP1](https://github.com/KRVP1)
- Мосунов Данил — [@dvmosunov](https://t.me/dvmosunov),  [Yunypd](https://github.com/yunypb)

**Куратор проекта**:
- Качкин Дмитрий — [@KachkinDmitrii](https://t.me/KachkinDmitrii), [Dmitry426](https://github.com/Dmitry426)

## 🛠️ Технические детали

- Python: 3.9+
- Используемые библиотеки: FastAPI, Streamlit, Scikit-learn, Pandas, etc.
- Контейнеризация: Docker, Docker-Compose.

## License

Проект распространяется по лицензии MIT
