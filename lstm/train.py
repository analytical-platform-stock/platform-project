import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def preparing_data(ticker, look_back, batch_size):
    scaler = MinMaxScaler(feature_range=(0, 1))

    ticker.loc[:, "time"] = pd.to_datetime(ticker.time).dt.date

    train_data = ticker[ticker.time < (ticker["time"].max() - pd.DateOffset(years=1)).date()]
    test_data  = ticker[ticker.time >= (ticker["time"].max() - pd.DateOffset(years=1)).date()]

    train_data = train_data.close.values.reshape(len(train_data), 1)
    test_data = test_data.close.values.reshape(len(test_data), 1)

    train_data = scaler.fit_transform(train_data)
    test_data = scaler.fit_transform(test_data)

    X_train, y_train = [], []
    # Create sequences and labels for training data
    for i in range(len(train_data) - look_back):
        X_train.append(train_data[i:i+look_back])
        y_train.append(train_data[i+look_back])

    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32)

    X_test, y_test = [], []
    # Create sequences and labels for training data
    for i in range(len(test_data) - look_back):
        X_test.append(test_data[i:i+look_back])
        y_test.append(test_data[i+look_back])

    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_test = torch.tensor(np.array(y_test), dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader, scaler


def train_val(model, loss_fn, optimizer, num_epoch, device, train_dataloader, test_dataloader, plot_=True):
    train_loss = []
    test_loss = []
    for epoch in range(num_epoch):
        model.train()
        loss_epoch = 0
        for batch_X, batch_y in train_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            pred = model(batch_X)
            loss = torch.sqrt(loss_fn(pred, batch_y))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_epoch += loss.item()

        train_loss.append(loss_epoch / len(train_dataloader))

        model.eval()
        with torch.no_grad():
            loss_epoch = 0
            for batch_X, batch_y in test_dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                pred = model(batch_X)
                loss = torch.sqrt(loss_fn(pred, batch_y))

                loss_epoch += loss.item()
            
            test_loss.append(loss_epoch / len(test_dataloader))
        
        # if (epoch + 1) % 10 == 0:
        #     print(f'Epoch [{epoch + 1}/{num_epoch}] - Training Loss: {train_loss[epoch]:.4f}, Test Loss: {test_loss[epoch]:.4f}\n')
    if plot_:
        plt.plot(train_loss, scalex=True, label="Training loss")
        plt.plot(test_loss, scalex=True, label="Test  loss")
        plt.legend()
        plt.show()


def test(model, device, test_dataloader, ticker, scaler, look_back, plot_=True):
    ticker_preds = []

    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            pred = model(batch_X)

            ticker_preds += [*pred.to("cpu")]
    
    ticker_preds = scaler.inverse_transform(ticker_preds)
    rmse = np.sqrt(mean_squared_error(ticker[ticker.time >= (ticker["time"].max() - pd.DateOffset(years=1)).date()].close[look_back:], ticker_preds))
    mae = mean_absolute_error(ticker[ticker.time >= (ticker["time"].max() - pd.DateOffset(years=1)).date()].close[look_back:], ticker_preds)
    mape = mean_absolute_percentage_error(ticker[ticker.time >= (ticker["time"].max() - pd.DateOffset(years=1)).date()].close[look_back:], ticker_preds)

    if plot_:
        plt.figure(figsize=(20,6))
        plt.plot(ticker[ticker.time >= (ticker["time"].max() - pd.DateOffset(years=1)).date()].time, 
                 ticker[ticker.time >= (ticker["time"].max() - pd.DateOffset(years=1)).date()]['close'], lw=2, label="close")
        plt.plot(ticker[ticker.time >= (ticker["time"].max() - pd.DateOffset(years=1)).date()].time[look_back:], ticker_preds.reshape(-1), lw=2, label="pred")
        plt.legend()
        plt.xlabel("date")
        plt.xlabel("date", fontsize=16)
        plt.ylabel("Close price", fontsize=16)
        plt.title(ticker.company.iloc[0], fontsize=16);
        plt.show()

        print(f"LSTM {ticker.company.iloc[0]} RMSE: {rmse}\nLSTM {ticker.company.iloc[0]} MAE: {mae}\nLSTM {ticker.company.iloc[0]} MAPE: {mape}")
    
    return rmse, mae, mape
