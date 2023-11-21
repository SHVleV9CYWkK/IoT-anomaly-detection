import pandas as pd
from sklearn.preprocessing import MinMaxScaler


if __name__ == '__main__':
    data = pd.read_csv('datasets/IoT_Modbus.csv')
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
    order = ['year', 'month', 'day', 'hour', 'minute', 'second', 'dayofweek', 'FC1_Read_Input_Register', 'FC2_Read_Discrete_Value', 'FC3_Read_Holding_Register', 'FC4_Read_Coil', 'label']
    data = data[order].astype('int32')

    # Calculate split points
    split_idx = int(len(data) * 0.8)

    # Split the data set, keeping order
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    # Separate features and labels
    X_train = train_data.drop('label', axis=1)
    y_train = train_data['label']
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']

    feature_columns = [col for col in X_train.columns if col != 'label']
    scaler = MinMaxScaler()
    X_train[feature_columns] = scaler.fit_transform(X_train[feature_columns]).astype('float32')
    X_test[feature_columns] = scaler.transform(X_test[feature_columns]).astype('float32')

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_data.to_csv('datasets/train_dataset.csv', index=False)
    test_data.to_csv('datasets/test_dataset.csv', index=False)

