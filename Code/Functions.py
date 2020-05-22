from PyQt5.QtWidgets import QFileDialog
import pandas as pd

def file_open():
    path = QFileDialog.getOpenFileName('Open CSV', )

def import_data():
    """
Imports csv or xls file from document explorer
    """
    filePath, _ = QFileDialog.getOpenFileName('Open file', '/home', filter="csv(*.csv)")
    if filePath != "":
        data = pd.read_csv(str(filePath))
    return data


def printer():
    print('hel')


def evaluate_forecast(y_pred, y_true, metric):
    """
Evaluate the performance of forecasted values against actual values according to some metric
    :param y_pred: vector
    :param y_true: vector
    :param metric: string (mape, mse)
    """
    if metric == 'mape':
        return 0
    if metric == 'mse':
        return 1
