import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Оценка модели ML
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    return accuracy, report, conf_matrix

# Оценка модели нейронной сети
def evaluate_neural_network(model, X_test, y_test):
    _, accuracy = model.evaluate(X_test, y_test)

    predictions = model.predict(X_test)

    pred_classes = np.argmax(predictions, axis=1)

    report = classification_report(y_test, pred_classes)
    conf_matrix = confusion_matrix(y_test, pred_classes)

    return accuracy, report, conf_matrix