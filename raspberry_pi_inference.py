import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from tqdm import tqdm


class LightweightLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, width_multiplier=1.0):
        super(LightweightLSTM, self).__init__()
        # Adjust hidden size based on the width multiplier
        adjusted_hidden_size = int(hidden_size * width_multiplier)

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, adjusted_hidden_size, num_layers=num_layers, batch_first=True)

        self.linear_1 = nn.Linear(adjusted_hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)

        # Take the output of the last time step
        last_time_step_out = lstm_out[:, -1, :]

        # Output layer
        x = self.linear_1(last_time_step_out)
        out = self.linear_2(x)
        return out


if __name__ == '__main__':
    seed = 42
    data = pd.read_csv('IoT_Modbus.csv')
    data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
    data['year'] = data['datetime'].dt.year
    data['month'] = data['datetime'].dt.month
    data['day'] = data['datetime'].dt.day
    data['hour'] = data['datetime'].dt.hour
    data['minute'] = data['datetime'].dt.minute
    data['second'] = data['datetime'].dt.second
    data['dayofweek'] = data['datetime'].dt.dayofweek

    # Sort the data by datetime
    data = data.sort_values(by='datetime')

    # Drop the original date, time, and timestamp columns
    data.drop(['date', 'time', 'datetime', 'type'], axis=1, inplace=True)

    # Adjust feature order
    order = ['year', 'month', 'day', 'hour', 'minute', 'second', 'dayofweek', 'FC1_Read_Input_Register',
             'FC2_Read_Discrete_Value', 'FC3_Read_Holding_Register', 'FC4_Read_Coil', 'label']
    data = data[order].astype('int32')

    # Calculate split points
    split_idx = int(len(data) * 0.8)

    # Split the data set, keeping order
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    # Separate features and labels
    X_train = train_data.drop('label', axis=1)
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label'].to_numpy()

    feature_columns = [col for col in X_train.columns if col != 'label']
    scaler = MinMaxScaler()
    X_train[feature_columns] = scaler.fit_transform(X_train[feature_columns]).astype('float32')
    X_test[feature_columns] = scaler.transform(X_test[feature_columns]).astype('float32')

    features_num = X_train.shape[1]
    hidden_neurons_num = 512
    output_neurons_num = 1
    lstm_num_layers = 2
    multiplier = 0.5

    model = LightweightLSTM(features_num, hidden_neurons_num, output_neurons_num, lstm_num_layers, multiplier)

    model.load_state_dict(torch.load('model_2023-11-19_17-16-41.pt'))

    model.eval()

    X_test_tensor = torch.tensor(X_test.to_numpy()).float().unsqueeze(1)

    predictions = []
    labels = []

    with torch.no_grad():
        pbar = tqdm(total=len(X_test_tensor))
        for i in range(len(X_test_tensor)):
            sample = X_test_tensor[i].unsqueeze(0)

            output = model(sample)
            probability = torch.sigmoid(output)
            prediction = (probability > 0.5).float()
            predictions.append(prediction.item())

            labels.append(y_test[i])
            pbar.update(1)
        pbar.close()

    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    print("Accuracy: ", acc, ", Precision: ", precision, ", Recall: ", recall, ", F1: ", f1)

