# Оглавление
1. [Описание проекта](#1-описание-проекта)
2. [Структура проекта](#2-структура-проекта)
3. [Модели машинного обучения и нейронные сети](#3-модели-машинного-обучения-и-нейронные-сети)
4. [Итоги и результаты](#4-итоги-и-результаты)
5. [Запуск примера](#5-запуск-примера)

# 1. Описание проекта
[[вернуться к оглавлению]](#оглавление)

**Arrhythmia detector** — это веб-приложение, которое использует машинное обучение для определения типа аритмии сердца (нормальные удары, наджелудочковые экстрасистолы, желудочковые экстрасистолы, фузионные удары, неклассифицируемые удары) на основе предоставленных данных ЭКГ. Приложение построено с использованием Streamlit и различных моделей машинного обучения для прогнозирования.

![image](https://github.com/user-attachments/assets/10298573-649e-4776-842d-fe9fadde59d3)

Что вы найдете внутри:
- Разведочный анализ данных (EDA)
- Предобработка данных
- Обучение моделей машинного обучения
- Веб-приложение на Streamlit

<br><br>


# 2. Структура проекта
[[вернуться к оглавлению]](#оглавление)

Проект имеет следующую структуру:

```plaintext
Heart-arrhythmia-classificator/
│
├── config/
│   └── config.yaml          # Конфигурационный файл с путями к данным и параметрами предобработки
│
├── data/
│   ├── raw/                 # Исходные данные
│   └── processed/           # Обработанные данные
│
├── models/                  # Обученные модели и графики
│
├── notebooks/
│   └── EDA.ipynb            # Jupyter Notebook с разведочным анализом данных (EDA)
│
├── src/
│   ├── main.py              # Основной скрипт для запуска пайплайна обработки данных и обучения моделей
│   ├── preprocessing.py      # Скрипт для предобработки данных
│   ├── evaluate.py          # Скрипт для оценки моделей
│   ├── plot_graphics.py     # Скрипт для построения графиков
│   ├── train.py             # Скрипт для обучения моделей
│   └── model.py             # Скрипт для создания и сохранения моделей
│
├── web/
│   ├── app.py               # Веб-приложение на Streamlit
│   └── utils.py             # Утилиты для веб-приложения
│
├── docs/                    # Отчеты и документация
│
├── requirements.txt         # Зависимости проекта
│
└── README.md                # Описание проекта
```

<br><br>

# 3. Модели машинного обучения и нейронные сети
[[вернуться к оглавлению]](#оглавление)

### 3.1. Полносвязная нейронная сеть (FNN)
Архитектура полносвязной нейронной сети (FNN) включает:
- Входной слой: 187 признаков
- Скрытые слои: полносвязный слой с 128 нейронами, полносвязный слой с 64 нейронами + Dropout (0.15) для регуляризации
- Выходной слой: Softmax на 5 классов

![fnn h5](https://github.com/user-attachments/assets/9eed0907-a31e-489f-b76a-1f253f0d70c5)

### 3.2. Сверточная нейронная сеть (CNN)
Архитектура сверточной нейронной сети (CNN) включает:
- Conv1D: сверточный слой с 32 фильтрами
- Dense: полносвязный слой с 512 нейронами
- BatchNormalization: нормализация батча для стабилизации обучения
- MaxPooling1D: снижение размерности признаков
- Dropout: регуляризация (0.5)
- Conv1D: сверточный слой с 64 фильтрами
- Dense: полносвязный слой с 1024 нейронами
- BatchNormalization и MaxPooling1D повторяются для глубокой обработки признаков
- Flatten: преобразование в вектор
-  Dense: выходной слой с softmax для классификации по 5 классам

![cnn h5](https://github.com/user-attachments/assets/42477cc0-f609-401f-9434-cb24b4724921)

### 3.3. Метод опорных векторов (SVM)
Модель метода опорных векторов (SVM):
- Использует ядро радиальной базисной функции (RBF) для нелинейной классификации.
![image](https://github.com/user-attachments/assets/ac798ab6-6a5f-4815-a357-f4e193b99eb5)

[Литература](https://scikit-learn.ru/stable/modules/gaussian_process.html#gp-kernels) 

### 3.4. Случайный лес (RFC)
Модель случайного леса:
- Состоит из 50 деревьев решений.

![image](https://github.com/user-attachments/assets/e740f995-d70b-493d-9a60-3b28b07d50cf)


### 3.5. K-ближайших соседей (KNC)
Модель K-ближайших соседей:
- Использует 11 соседей для классификации.
- Простая и эффективная для задач с небольшим количеством признаков.

![image](https://github.com/user-attachments/assets/f138f66b-6bb5-4a3e-86a0-4b4820ed0ab9)


### 3.6. XGBoost (XGB)
Модель XGBoost:
- Использует градиентный бустинг для улучшения точности классификации.
- Автоматически настраивает параметры для достижения оптимальной производительности.

![image](https://github.com/user-attachments/assets/2f1408e5-dcb6-42e3-84fe-711cf474cce1)


# 4. Итоги и результаты
[[вернуться к оглавлению]](#оглавление)

### 4.1. Набор данных
Данный набор данных был взят с Kaggle ([Источник](https://www.kaggle.com/datasets/shayanfazeli/heartbeat/data)).

Описание из оригинального источника:
> Этот набор данных состоит из двух коллекций сигналов ЭКГ, полученных из известного набора данных по классификации сердечных ритмов: базы данных аритмий MIT-BIH ([Источник](https://www.physionet.org/content/mitdb/1.0.0/)). Общее количество образцов в обеих коллекциях достаточно велико для обучения глубокой нейронной сети. Сигналы соответствуют электрокардиограммам (ЭКГ) сердечных сокращений как в нормальном состоянии, так и при различных аритмиях. Эти сигналы предварительно обработаны и сегментированы, причем каждый сегмент соответствует одному сердечному сокращению.

Вы можете увидеть EDA в **notebooks/EDA.ipynb**.
<br><br>
### 4.2. Результаты обучения и оценки
[[вернуться к оглавлению]](#оглавление)

- После обучения моделей мы достигли следующих результатов:

| Модель                     | Accuracy |
|----------------------------|----------|
| SVC (Support Vector Classifier) | 0.9054   |
| RFC (Random Forest Classifier) | 0.9812   |
| KNC (K-Nearest Neighbors Classifier) | 0.9050   |
| XGB (XGBoost)               | 0.9728   |
| FNN (Feedforward Neural Network) | 0.9638   |
| CNN (Convolutional Neural Network) | 0.9570   |


- Как видно выше, все результаты очень хорошие.
<br><br>
### 4.3. Использование модели
[[вернуться к оглавлению]](#оглавление)

Финальные модели будут использоваться в виде веб-сервиса на Streamlit через загрузку файлов форматов **.h5** и **.joblib**.

- Приложение принимает данные ЭКГ, обрабатывает их, делает предсказание и возвращает результат.
- Пользователь может выбрать загрузить свои данные или использовать ползунок для выбора записи из тестовой выборки.
- При выборе записи из тестовой выборки строится график Plotly, который можно двигать и масштабировать.
- Пользователю предлагается список моделей на выбор.
- После нажатия кнопки "Predict" выводится таблица с точностью модели на валидационной выборке, предсказанным классом и истинным классом.
- Если пользователь загружает данные в формате .csv, они должны быть содержать одну строку с 187 столбцами.
- Если в данных содержится меньше 187 столбцов, то можно дополнить нулями до 187 столбцов.
- Данные проходят предобработку, включая фильтрацию с использованием фильтра нижних частот Баттерворта и нормализацию, после чего загружаются в модель.
- На выбор доступны модели машинного обучения и нейронные сети (FNN, CNN).

<br><br>

# 5. Запуск примера
[[вернуться к оглавлению]](#оглавление)

### 5.1. Настройка
- **Скачайте содержимое этого репозитория**

  Вы можете либо клонировать этот репозиторий, либо просто скачать его и распаковать в какую-нибудь папку:

 ```bash
  git clone https://github.com/syndicate2k/Heart-arrhythmia-classificator
  cd Heart-arrhythmia-classificator
  ```
- **Установите зависимости**

  Установите необходимые библиотеки с помощью pip:

  ```
  pip install -r requirements.txt
  ```

- **Запустите приложение**

  Приложение будет доступно по адресу http://127.0.0.1:8501/ после выполнения команды:

  ```
  streamlit run web/app.py
  ```
- **Инструкция по использованию**
  * Откройте приложение в браузере.
  * Загрузите свои данные ЭКГ или выберите запись из тестовой выборки с помощью ползунка.
  * При использовании ползунка отобразится интерактивный график Plotly.
  * Выберите модель из списка.
  * Нажмите "Predict" для получения предсказания.
