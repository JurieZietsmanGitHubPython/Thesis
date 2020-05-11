import sys
import os
import pandas as pd
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from Functions import import_csv


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("User Interface/main_window.ui", self)

        self.pushButton1.clicked.connect(import_csv)


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()
