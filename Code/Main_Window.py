import sys
import os
import pandas as pd
import numpy as np
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QHeaderView, QApplication
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from Functions import MainWindow
import warnings
import qdarkgraystyle

warnings.filterwarnings("ignore")


class Gui_Application(MainWindow):

    def __init__(self):
        MainWindow.__init__(self)

        ############################################################
        # Threads
        ############################################################

        ############################################################
        # Signals for startup, homepage, navigation and closing application
        ############################################################
        self.stack.setCurrentIndex(0)
        self.orderListTabWidget.setCurrentIndex(0)
        self.localOrderTableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.importOrderTableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.startButton.clicked.connect(self.on_startButton_clicked)
        self.forecastingButton.clicked.connect(self.nextStack)
        self.orderanalysisButton.clicked.connect(self.nextStack)
        self.orderlistButton.clicked.connect(self.nextStack)
        self.quitButton.clicked.connect(self.close_application)

        ############################################################
        # Signals for data processing
        ############################################################
        self.historicSales = None
        self.skuList = None
        self.stockOnHand = None
        self.bomList = None
        self.supplierOrigin = None
        self.warehouseInfo = None
        self.backlogData = None

        self.historicSalesButton.clicked.connect(lambda: self.import_csv("historicSales"))
        self.skuListButton.clicked.connect(lambda: self.import_csv("skuList"))
        self.stockOnHandButton.clicked.connect(lambda: self.import_csv("stockOnHand"))
        self.bomListButton.clicked.connect(lambda: self.import_csv("bomList"))
        self.supplierOriginButton.clicked.connect(lambda: self.import_csv("supplierOrigin"))
        self.warehouseInfoButton.clicked.connect(lambda: self.import_csv("warehouseInfo"))
        self.backlogDataButton.clicked.connect(lambda: self.import_csv("backlogData"))

        self.testButton.clicked.connect(self.print_import)

        ############################################################
        # Signals for forecasting
        ############################################################

        ############################################################
        # Signals for order analysis
        ############################################################

        ############################################################
        # Signals for order list
        ############################################################
        self.orderlistButton.clicked.connect(self.createTable)
        self.confirmlocalOrdersButton.clicked.connect(self.createTable)  # Just for testing

        self.checkFeasibilityButton.clicked.connect(self.clearTable)


class Worker(QObject):
    hashed = pyqtSignal(str, str)

    @pyqtSlot(str)
    def hash_directory(self, root):
        hash_gen = recursive_hashes(Path(root))
        for path, sha1_hash in hash_gen:
            self.hashed.emit(path, sha1_hash)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Gui_Application()
    app.setStyleSheet(qdarkgraystyle.load_stylesheet())
    window.show()
    sys.exit(app.exec_())
