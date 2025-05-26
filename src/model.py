import xgboost as xgb
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from joblib import dump

# Создание модели по имени
def create_model(model_name):
    if model_name == 'RFC':
        return RFC()
    if model_name == 'SVC':
        return SVC()
    if model_name == 'KNC':
        return KNC()
    if model_name == 'XGB':
        return XGB()
    if model_name == 'FNN':
        return FNN()
    if model_name == 'CNN':
        return CNN()
    else:
        raise ValueError("Unknown model name")

# Сохранение в файл
def save_ml_model(model, model_path):
    dump(model, model_path)

# Сохранение в файл
def save_neural_network(model, model_path):
    save_model(model, model_path)

# SVM с радиальной базисной функцией
def SVC():
    model = svm.SVC(kernel='rbf', random_state=42)
    return model

# Случайный лес
def RFC():
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    return model

# K-ближайших соседей
def KNC():
    model = KNeighborsClassifier(n_neighbors=11)
    return model

# XGBoost
def XGB():
    model = xgb.XGBClassifier()
    return model

# Полносвязная нейронная сеть
def FNN():
    '''
    Архитектура:
      - Входной слой: 187 признаков
      - Скрытые слои: полносвязный слой с 128 нейронами, полносвязный слой с 64 нейронами + Dropout (0.15) для регуляризации
      - Выходной слой: Softmax на 5 классов
     '''
    model = Sequential([
        Dense(128, activation='relu', input_shape=(187,)),
        Dropout(0.15),
        Dense(64, activation='relu'),
        Dropout(0.15),
        Dense(5, activation='softmax') 
    ])
    return model

# Сверточная нейронная сеть
def CNN():
    '''
     Архитектура:
       - Conv1D: сверточный слой с 32 фильтрами
       - Dense: полносвязный слой с 512 нейронами
       - BatchNormalization: нормализация батча для стабилизации обучения
       - MaxPooling1D: снижение размерности признаков
       - Dropout: регуляризация (0.5)
       - Conv1D: сверточный слой с 64 фильтрами
       - Dense: полносвязный слой с 1024 нейронами
       - BatchNormalization и MaxPooling1D повторяются для глубокой обработки признаков
       - Flatten: преобразование в вектор 
       - Dense: выходной слой с softmax для классификации по 5 классам
     '''
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=(187,1)),
        Dense(512, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),
        Conv1D(64, kernel_size=3, activation='relu'),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(5, activation='softmax') 
    ])
    return model