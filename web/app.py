import sys
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
import yaml
from joblib import load
from tensorflow.keras.models import load_model
from utils import predict_sample, get_model_accuracy, sample_test_classes

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_data, split_data, normalize_data

# Загрузка конфигурации
with open('config/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Загрузка данных
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, header=None)

# Загрузка моделей
@st.cache_resource
def load_models(model_dir):
    models = {}
    for model_file in os.listdir(model_dir):
        if model_file.endswith('.joblib'):
            model_path = os.path.join(model_dir, model_file)
            models[model_file] = load(model_path)
        elif model_file.endswith('.h5'):
            model_path = os.path.join(model_dir, model_file)
            models[model_file] = load_model(model_path)
    return models

# Функция для построения графика сигнала ЭКГ
def plot_signal_with_plotly(X):
    # Создаем фигуру
    fig = go.Figure()

    # Добавляем график на фигуру
    fig.add_trace(go.Scatter(
        y=X.iloc[0],
        mode='lines'
    ))

    # Обновляем макет фигуры
    fig.update_layout(
        title='Сигнал ЭКГ',
        xaxis_title='Индекс',
        yaxis_title='Значение',
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='darkblue')
    )

    # Отображение
    st.plotly_chart(fig)

# Основная функция
def main():
    # Настройка страницы
    st.set_page_config(page_title="Arrhythmia Detector", layout="wide")

    # CSS стили
    st.markdown("""
    <style>
    .main {
        background-color: #e6f3ff;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .stSelectbox, .stSlider {
        font-family: 'Arial', sans-serif;
    }
    .stMarkdown {
        font-family: 'Arial', sans-serif;
    }
    hr.style {
        border: 0;
        height: 1px;
        background: #333;
        background-image: linear-gradient(to right, #ccc, #333, #ccc);
    }
    .column {
        border-right: 1px solid #ccc;
    }
    .column:last-child {
        border-right: none;
    }
    .title {
        font-size: 2.5em !important;
        text-align: center !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Заголовок
    st.markdown("<h1 class='title'>Проверка ЭКГ на наличие аритмии ❤️🩺</h1>", unsafe_allow_html=True)

    # Добавление линии разграничителя
    st.markdown("<hr class='hr-style'>", unsafe_allow_html=True)

    # Выбор источника данных
    data_source = st.radio("Выберите источник данных:", ("Использовать тестовые данные", "Загрузить файл"))

    if data_source == "Использовать тестовые данные":
        data = sample_test_classes(load_data(config['data']['test_data_path']))
        show_slider = True
    else:
        uploaded_file = st.file_uploader("Загрузите файл с данными ЭКГ", type=["csv"])
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            if data.shape[1] != 187:
                st.error("Файл должен содержать 187 столбцов.")
                return
            show_slider = False
        else:
            st.warning("Пожалуйста, загрузите файл с данными.")
            return

    # Метки классов для интерпретации пред. класса
    class_labels = config['class_labels']

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="column">', unsafe_allow_html=True)
        st.header("Визуализация сигнала")

        # Слайдер
        if show_slider:
            row_index = st.slider("Выберите номер записи", 1, len(data), 1)
        else:
            row_index = 1
        
        row_index -= 1

        if row_index < len(data):
            row_data = data.iloc[row_index]
            if data_source == "Использовать тестовые данные":
                X, y = split_data(pd.DataFrame([row_data]))
            else:
                X = pd.DataFrame([row_data])
                y = None  # костыль 

            plot_signal_with_plotly(X)
        else:
            st.error("Выбранный индекс строки выходит за пределы доступных данных.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="column">', unsafe_allow_html=True)
        st.header("Прогнозирование")

        # Загрузка моделей
        models = load_models(config['models']['model_dir'])

        # Преобразуем имена файлов для отображения в нормальном виде
        model_files = [file.split('.')[0].upper() for file in config['models']['model_files']]

        # Выбор модели
        model_file_display = st.selectbox("Выберите модель", options=model_files)

        # Получаем имя выбранного файла
        model_file = config['models']['model_files'][model_files.index(model_file_display)]

        model = models[model_file]

        # Костыль, парсим имя модели до точки
        model_name = model_file.split('.')[0]

        # Предобработка данных
        scaler = load(config['preprocessing']['scaler_path'])
        X_normalized, _ = normalize_data(X, scaler)

        # Предикт
        if st.button("Predict"):
            y_pred, y_true = predict_sample(model, X_normalized, y)
            accuracy = get_model_accuracy(model_name, config)

            # Создание таблицы с результатами
            results = pd.DataFrame({
                "Метрика": ["Точность модели на валидационной выборке", "Предположительный класс заболевания", "Действительный класс заболевания"],
                "Значение": [
                    f"{accuracy * 100:.2f}%",
                    class_labels[y_pred],
                    class_labels[y_true] if y_true is not None else "N/A"
                ]
            })

            # Рисуем таблицу
            st.table(results)
        st.markdown('</div>', unsafe_allow_html=True)

    # Гор. линия внизу
    st.markdown("<hr class='hr-style'>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()