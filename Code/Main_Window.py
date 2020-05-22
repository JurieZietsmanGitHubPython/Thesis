import sys
import os
import pandas as pd
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from Functions import import_data, printer
import warnings
warnings.filterwarnings("ignore")


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("C:/Users/Jurie/PycharmProjects/Thesis/User Interface/main_window.ui", self)

        self.import_csv_button.clicked.connect(printer)

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()
