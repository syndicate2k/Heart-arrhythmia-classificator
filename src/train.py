from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from src.model import create_model
from src.evaluate import evaluate_model, evaluate_neural_network

# Обучение модели ML
def train_model(X, y, model_name):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

    model = create_model(model_name)
    model.fit(X_train, y_train)

    accuracy, report, conf_matrix = evaluate_model(model, X_valid, y_valid)

    return model, accuracy, report, conf_matrix

# Обучение нейронной сети
def train_neural_network(X, y, model_name, epochs=1, batch_size=128):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

    model = create_model(model_name)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epochs, batch_size=batch_size, verbose = 1, callbacks=[early_stopping])
    
    accuracy, report, conf_matrix = evaluate_neural_network(model, X_valid, y_valid)

    return model, history, accuracy, report, conf_matrix



