# Import required libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
from keras import Sequential
from keras.src.layers import Dense, Input
from keras.src.callbacks import EarlyStopping
from keras.src.optimizers import SGD
from keras.src.regularizers import L2


def create_model(input_shape, hidden_units, learning_rate, momentum, l2_lambda):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(hidden_units, activation='elu',
              kernel_regularizer=L2(l2_lambda)),
        Dense(1, activation='sigmoid')
    ])

    optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'binary_crossentropy', 'mean_squared_error']
    )
    return model


def plot_convergence(histories, lr, momentum, l2_lambda):
    plt.figure(figsize=(10, 6))
    min_epochs = min(len(h.history['loss']) for h in histories)
    train_losses = np.array([h.history['loss'][:min_epochs]
                            for h in histories])
    val_losses = np.array([h.history['val_loss'][:min_epochs]
                          for h in histories])

    avg_train_loss = np.mean(train_losses, axis=0)
    avg_val_loss = np.mean(val_losses, axis=0)

    plt.plot(avg_train_loss, label='Train Loss')
    plt.plot(avg_val_loss, label='Validation Loss')
    plt.title(
        f'Average Loss Convergence (lr={lr}, m={momentum}, l2={l2_lambda})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'plots/convergence_lr{lr}_m{momentum}_l2{l2_lambda}.png')
    plt.close()


def plot_feature_importance(model, feature_names, mode):
    # Get weights from first layer
    first_layer_weights = model.layers[0].get_weights()[0]

    # Calculate importance as mean absolute weight
    feature_importance = np.mean(np.abs(first_layer_weights), axis=1)

    if (mode == 1):
        plt.figure(figsize=(10, 6))
        plt.bar(feature_names, feature_importance,
                color='skyblue', edgecolor='black')
        plt.title('Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        importance_list = feature_importance.tolist()
        feature_list = list(feature_names)

        # Create pairs and sort
        feature_pairs = list(zip(feature_list, importance_list))
        feature_pairs.sort(key=lambda x: x[1], reverse=True)

        # Get top 10
        top_10 = feature_pairs[:10]
        top_features = [pair[0] for pair in top_10]
        top_importance = [pair[1] for pair in top_10]

        plt.figure(figsize=(10, 6))
        plt.bar(top_features, top_importance,
                color='lightcoral', edgecolor='black')
        plt.title('Top 10 Most Important Features')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


def train_NN_full_dataset(mode=0):
    # Fixed hyperparameters
    hidden_units = 76
    learning_rate = 0.05
    momentum = 0.6
    l2_lambda = 0.0001

    FILE = "alzheimers_disease_data.csv"

    try:
        df = pd.read_csv(FILE, encoding="utf-8")
        print("File loaded successfully")

        # Preprocessing
        columns_for_normalization = ['AlcoholConsumption', 'PhysicalActivity',
                                     'DietQuality', 'SleepQuality', 'ADL']
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[columns_for_normalization] = scaler.fit_transform(
            df[columns_for_normalization])

        columns_for_standardization = ['Age', 'BMI', 'SystolicBP', 'DiastolicBP',
                                       'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
                                       'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment']
        scaler_standard = StandardScaler()
        df[columns_for_standardization] = scaler_standard.fit_transform(
            df[columns_for_standardization])

        df["Ethnicity"] = df["Ethnicity"].astype("category")
        df["EducationLevel"] = df["EducationLevel"].astype("category")
        df = pd.get_dummies(df, columns=["Ethnicity"], prefix="Ethnicity")
        df = pd.get_dummies(
            df, columns=["EducationLevel"], prefix="EducationLevel")

        # mode = 0 trains NN with whole dataset and mode = 1 trains with the features selected by the GA
        if (mode == 0):
            dataForTargetColumn = df.drop(
                columns=['PatientID', 'Diagnosis', 'DoctorInCharge'])
            selected_features = dataForTargetColumn
        else:
            selected_features = ['SleepQuality', 'CardiovascularDisease', 'CholesterolLDL', 'MMSE',
                                 'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL']

            dataForTargetColumn = df[selected_features]

        targetColumn = df['Diagnosis']
        dataForTargetColumn = dataForTargetColumn.to_numpy().astype('float32')
        targetColumn = targetColumn.to_numpy().astype('float32')

        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        fold_losses = []
        fold_accuracies = []
        val_accuracies = []
        fold_bces = []
        fold_mces = []
        histories = []

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            min_delta=0.001,
            restore_best_weights=True,
            mode='min',
            verbose=1
        )

        for train_idx, val_idx in skf.split(dataForTargetColumn, targetColumn):
            x_train, x_val = dataForTargetColumn[train_idx], dataForTargetColumn[val_idx]
            y_train, y_val = targetColumn[train_idx], targetColumn[val_idx]
            y_train = y_train.reshape(-1, 1)
            y_val = y_val.reshape(-1, 1)

            model = create_model(dataForTargetColumn.shape[1], hidden_units,
                                 learning_rate, momentum, l2_lambda)

            history = model.fit(
                x_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(x_val, y_val),
                verbose=0,
                callbacks=[early_stopping]
            )

            train_metrics = model.evaluate(x_train, y_train, verbose=0)
            val_metrics = model.evaluate(x_val, y_val, verbose=0)

            fold_losses.append(train_metrics[0])
            fold_accuracies.append(train_metrics[1])
            fold_bces.append(train_metrics[2])
            fold_mces.append(train_metrics[3])
            val_accuracies.append(val_metrics[1])
            histories.append(history)

        if (mode == 1):
            plot_convergence(histories, learning_rate, momentum, l2_lambda)

        results = {
            'Train Accuracy': np.mean(fold_accuracies),
            'Train Loss': np.mean(fold_losses),
            'Train BCE': np.mean(fold_bces),
            'Train MSE': np.mean(fold_mces),
            'Validation Accuracy': np.mean(val_accuracies)
        }

        plot_feature_importance(model, selected_features, mode=mode)

        for res in results:
            print(f"{res}: {results[res]:.4f}")

        return model
    except Exception as e:
        print(f"❌ An error occurred: {e}")


def main():
    train_NN_full_dataset()


if __name__ == "__main__":
    main()
