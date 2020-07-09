from PyQt5.QtCore import pyqtSlot as Slot
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem, QMainWindow
from PyQt5 import uic, Qt, QtGui

import pandas as pd
import sys
import sktime
import ui_background

from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import smape_loss
import numpy as np


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("C:/Users/Jurie/PycharmProjects/Thesis/User Interface/main_window.ui", self)
        self.setWindowIcon(QtGui.QIcon('Images for QT/Black_dot.png'))

    ##############################################################
    # Functions for testing
    ##############################################################

    def change_text(self):
        self.label_3.setText('lekker')

    # def test(self, df):
    #     print(df)
    #     print(type(df))

    # @Slot()
    def printer(self, dataframe):
        print(dataframe)

    ##############################################################
    # Functions for homepage, navigation and closing application
    ##############################################################
    #@Slot()
    def on_startButton_clicked(self):
        self.stack.setCurrentIndex(1)

    def nextStack(self):
        index = self.stack.currentIndex()
        index = index + 1
        self.stack.setCurrentIndex(index)

    def close_application(self):
        choice = QMessageBox.question(self, 'Close application', "Quit the application?",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if choice == QMessageBox.Yes:
            sys.exit()
        else:
            pass

    ##############################################################
    # Functions for data processing
    ##############################################################
    #@Slot(str)
    def import_csv(self, button):
        """
    Reads csv file from document explorer
        """
        filePath, _ = QFileDialog.getOpenFileName(self, 'Choose a file', '\Home', filter="csv(*.csv)")
        if filePath != "":
            data = pd.read_csv(str(filePath))

        if button == "historicSales":
            data['POSTING DATE'] = pd.to_datetime(data['POSTING DATE'].astype(str), format='%Y%m%d')

            column_remove = ['BRANCH', 'CUSTNO', 'DIN', 'MATL', 'ITEM NO', 'WEIGHT',
                             'USER NAME', 'TYPEOFSALE', 'SLM', 'INVOICE NUMBER', 'NAME', '.',
                             'THREAD', 'PL', 'COSTVAL', 'SALES VALUE', 'OVERBY', 'DESCRIPT',
                             'INDUSTRY CODE', 'CONTRIBUTION%', 'GLGROUP', 'TRNSPCODE',
                             ' ']
            data.drop(column_remove, axis=1, inplace=True)
            data = data.groupby(['POSTING DATE', 'STOCKNO']).sum()

            self.historicSales = data

        if button == "skuList":
            data = data['STOCKNO'].tolist()
            self.skuList = data

        if button == "stockOnHand":
            self.stockOnHand = data

        if button == "bomList":
            self.bomList = data

        if button == "supplierOrigin":
            self.supplierOrigin = data

        if button == "warehouseInfo":
            self.warehouseInfo = data

        if button == "backlogData":
            self.backlogData = data

    def print_import(self):
        print(self.historicSales)
        print(type(self.historicSales))

    ##############################################################
    # Functions for forecasting
    ##############################################################

    def perform_forecast(self, skuList, historicSales):
        """
        Perform forecast for each SKU in skuList
        """
        pass

        historicDataUsed = self.historicUsageSpinBox.value()  # Get number of years of data to use
        forecastingPeriod = self.forecastPeriodSpinBox.value()  # Get forecasting period to use
        forecastingPeriodType = self.periodTypeComboBox.value()  # Get forecasting period type to use

        for sku in skuList:
            skuHistoric = historicSales[historicSales.index.isin([sku], level='STOCKNO')]
            skuHistoric.index = skuHistoric.index.droplevel('STOCKNO')
            skuHistoric = skuHistoric.resample('M').sum()
            print(skuHistoric)


    def evaluate_forecast(self, y_pred, y_true, metric):
        """
        Evaluate the performance of forecasted values against actual values according to some metric
        :param y_pred: vector
        :param y_true: vector
        :param metric: string (mape, mse)
        """
        pass
        if metric == 'mape':
            return 0
        if metric == 'mse':
            return 1

    ##############################################################
    # Functions for order analysis
    ##############################################################

    ##############################################################
    # Functions for order list
    ##############################################################

    def createTable(self):
        """
        Populate table widgets to list suggested local and import orders
        """
        # Set number of rows
        self.localOrderTableWidget.setRowCount(2)
        self.importOrderTableWidget.setRowCount(2)

        # Populate rows
        self.localOrderTableWidget.setItem(0, 0, QTableWidgetItem("0099"))
        self.localOrderTableWidget.setItem(1, 0, QTableWidgetItem("0022"))

        self.importOrderTableWidget.setItem(0, 0, QTableWidgetItem("0066"))
        self.importOrderTableWidget.setItem(1, 0, QTableWidgetItem("0088"))

    def clearTable(self):
        """
        Clear table widget from current contents
        """
        while self.localOrderTableWidget.rowCount() > 0:
            self.localOrderTableWidget.removeRow(0)

        while self.importOrderTableWidget.rowCount() > 0:
            self.importOrderTableWidget.removeRow(0)
