import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Предикт с использованием готовой модели для streamlit
def predict_sample(model, X, y=None):
    y_pred = model.predict(X)

    if hasattr(model, 'layers'):
        pred_class = int(np.argmax(y_pred[0]))
    else:
        pred_class = int(y_pred[0])

    # y=None если мы сами загружаем файл
    if y is not None:
        true_class = int(y.iloc[0])
    else:
        true_class = None

    return pred_class, true_class

# Получение точности модели на валид. выборки при обучении
def get_model_accuracy(model_name, config):

    # Костыль / FIX LATER
    model_name = model_name.lower()

    accuracy_df = pd.read_csv(config['models']['models_accuracy_path'])

    # Костыль / FIX LATER
    accuracy_df['Model'] = accuracy_df['Model'].str.strip().str.lower()

    model_accuracy = accuracy_df.loc[accuracy_df['Model'] == model_name, 'Accuracy'].values

    return model_accuracy[0]

# Делаем случайную выборку из тестовых данных
def sample_test_classes(df, num_samples=15):
    # Получаем уникальные классы из последнего столбца
    classes = df.iloc[:, -1].unique()

    sampled_df = pd.DataFrame()

    # Для каждого класса берем указанное количество строк
    for cls in classes:
        cls_samples = df[df.iloc[:, -1] == cls].head(num_samples)
        sampled_df = pd.concat([sampled_df, cls_samples])

    # Тусуем
    shuffled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return shuffled_df