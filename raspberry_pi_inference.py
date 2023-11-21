import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
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

        self.fc = nn.Linear(adjusted_hidden_size, output_size)

    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)

        # Take the output of the last time step
        last_time_step_out = lstm_out[:, -1, :]

        # Output layer
        out = self.fc(last_time_step_out)
        return out


if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser(description='Run LSTM model with optional quantization.')
    parser.add_argument('--quantize', action='store_true', help='Apply quantization to the model')
    args = parser.parse_args()

    train_data = pd.read_csv('train_dataset.csv')
    test_data = pd.read_csv('test_dataset.csv')

    # Separate the features and labels
    X_train = train_data.drop('label', axis=1).astype('float32')
    y_train = train_data['label']
    X_test = test_data.drop('label', axis=1).astype('float32')
    y_test = test_data['label']

    features_num = X_train.shape[1]
    hidden_neurons_num = 512
    output_neurons_num = 1
    lstm_num_layers = 2
    multiplier = 0.5

    model = LightweightLSTM(features_num, hidden_neurons_num, output_neurons_num, lstm_num_layers, multiplier)

    model.load_state_dict(torch.load('model_2023-11-20_00-57-22.pt'))

    model.eval()

    if args.quantize:
        print("Applying quantization")
        torch.backends.quantized.engine = 'qnnpack'
        model = torch.ao.quantization.quantize_dynamic(model, {nn.LSTM, nn.Linear}, dtype=torch.qint8)

    X_test_tensor = torch.tensor(X_test.to_numpy()).float().unsqueeze(1)

    predictions = []
    labels = []

    with torch.no_grad():
        pbar = tqdm(total=len(X_test_tensor))
        positive_samples_detected = 0
        for i in range(len(X_test_tensor)):
            sample = X_test_tensor[i].unsqueeze(0)

            output = model(sample)
            probability = torch.sigmoid(output)
            prediction = (probability > 0.5).float()
            if prediction.item() == 1:
                positive_samples_detected += 1
                pbar.set_description(f"Detected positive samples: {positive_samples_detected}")


            predictions.append(prediction.item())

            labels.append(y_test[i])
            pbar.update(1)
        pbar.close()

    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    print("\nAccuracy: ", acc, ", Precision: ", precision, ", Recall: ", recall, ", F1: ", f1)

