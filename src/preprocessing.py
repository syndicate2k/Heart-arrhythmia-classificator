import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Загрузка данных
def load_data(file_path):
    data = pd.read_csv(file_path)

    return data

# Фильтр нижних частот Баттерворта
def butter_lowpass(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)

    return filtered_data

# Ресемплим классы для устранения дизбаланса (см. EDA)
def resample_data(data):
    resampled_class_df = []

    # Фиксируем таргет
    target_column_index = data.shape[1] - 1

    # Обрабатываем каждый класс
    for class_label in range(5):
        # Разделяем данные на классы
        df_class = data[data.iloc[:, target_column_index] == class_label]

        # Применяем upsampling для каждого класса, кроме 0
        if class_label == 0:
            df_resampled = df_class.sample(n=5000, random_state=42)
        else:
            df_resampled = resample(df_class, replace=True, n_samples=5000, random_state=42)

        # Добавляем ресемплированный df в список
        resampled_class_df.append(df_resampled)

    # Объединяем все в один
    df_resampled = pd.concat(resampled_class_df)

    # Перемешиваем итоговые данные
    df_resampled = df_resampled.sample(frac=1, random_state=42).reset_index(drop=True)

    # Сохраняем в csv
    df_resampled.to_csv('data/processed/train_resampled.csv', index=False)
    
    return df_resampled

# Создание стандартизатора
def fit_scaler(data):
    scaler = StandardScaler()
    scaler.fit(data)

    return scaler

# Нормализация данных
def normalize_data(X, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(X)
    else:
        data_normalized = scaler.transform(X)
    return data_normalized, scaler

# Разделение датасета на признаки и таргет
def split_data(data):
    # Фиксируем таргет
    target_column_index = data.shape[1] - 1

    X = data.iloc[:, :target_column_index]  
    y = data.iloc[:, target_column_index]  

    return X, y
