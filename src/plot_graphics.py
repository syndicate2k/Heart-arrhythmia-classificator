import matplotlib.pyplot as plt
import seaborn as sns

# Accuracy + Loss
def plot_accuracy(history):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    plt.title('Model Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.title('Model Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.tight_layout()

# Матрица ошибок
def plot_conf_matrix(conf_matrix):
    plt.figure(figsize=(6, 5))

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

# Сигнал ЭКГ
def plot_signal(signal):
    plt.figure(figsize=(10, 8))
    plt.plot(signal) 

    plt.xlabel('Индекс')
    plt.ylabel('Значение')

    plt.grid(True)
    plt.tight_layout()
    