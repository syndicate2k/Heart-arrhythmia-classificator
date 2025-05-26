import sys
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump

# Добавление корневой директории в путь поиска модулей
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_data, resample_data, butter_lowpass, split_data, fit_scaler, normalize_data
from src.evaluate import evaluate_model, evaluate_neural_network
from src.plot_graphics import plot_conf_matrix, plot_accuracy
from src.train import train_model, train_neural_network
from src.model import save_ml_model, save_neural_network, create_model

# Загрузка конфигурации
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class Pipeline:
    def __init__(self, config):
        self.config = config

    def load_and_preprocess_data(self):
        # Загрузка данных
        data = load_data(self.config['data']['train_data_path'])

        # Ресемплинг
        resampled_data = resample_data(data)

        # Разделение данных на выборку и таргет
        X, y = split_data(resampled_data)

        # Применение низкочастотного фильтра Баттерворта
        X_filtered = butter_lowpass(X, self.config['preprocessing']['lowpass_cutoff'], self.config['preprocessing']['resample_rate'])

        # Масштабирование данных
        X_normalized, scaler = normalize_data(X_filtered)

        dump(scaler, config['preprocessing']['scaler_path'])

        return X_normalized, y

    def run_pipeline(self):
        X_normalized, y = self.load_and_preprocess_data()

        print("Данные предобработаны.")

        # Создание DataFrame для хранения точности моделей
        accuracy_df = pd.DataFrame(columns=['Model', 'Accuracy'])

        # NEED FIX
        models = {
            "SVC": "svc",
            "RFC": "rfc",
            "KNC": "knc",
            "XGB": "xgb",
            "FNN": "fnn",
            "CNN": "cnn"
        }

        for model_name, model_path in models.items():
            if model_name in ["FNN", "CNN"]:
                model, history, accuracy, report, conf_matrix = train_neural_network(X_normalized, y, model_name)
                save_neural_network(model, os.path.join(self.config['models']['model_dir'], f"{model_path}.h5"))

                # Сохранение графика точности
                plt.figure()
                plot_accuracy(history)
                plt.savefig(os.path.join(self.config['models']['model_dir'], f'{model_name}_accuracy.png'))
                plt.close()
            else:
                model, accuracy, report, conf_matrix = train_model(X_normalized, y, model_name)
                save_ml_model(model, os.path.join(self.config['models']['model_dir'], f"{model_path}.joblib"))

            # Сохранение графика матрицы ошибок
            plt.figure()
            plot_conf_matrix(conf_matrix)
            plt.savefig(os.path.join(self.config['models']['model_dir'], f'{model_name}_conf_matrix.png'))
            plt.close()

            new_row = pd.DataFrame([{'Model': model_name.lower(), 'Accuracy': accuracy}])
            # Объединение
            accuracy_df = pd.concat([accuracy_df, new_row], ignore_index=True)

            print(f"Модель {model_name} обучена и сохранена.")
            print(f"Accuracy: {accuracy}")
            print("Classification Report:")
            print(report)
            print("Confusion Matrix:")
            print(conf_matrix)

        # Сохранение DataFrame с точностью всех моделей в файл CSV
        accuracy_df.to_csv(os.path.join(self.config['models']['model_dir'], 'models_accuracy.csv'), index=False)

if __name__ == "__main__":
    pipeline = Pipeline(config)
    pipeline.run_pipeline()
