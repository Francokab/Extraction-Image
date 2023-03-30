from PyQt5.QtCore import QDateTime, Qt, QTimer, pyqtSignal,  pyqtSlot, QSize
from PyQt5.QtWidgets import (QMainWindow, QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QScrollArea)
from qt_material import apply_stylesheet
import sys
from ProcessTab import ProcessTab

class MainGUI(QTabWidget):
    updateDatabaseSignal = pyqtSignal(list)

    def __init__(self, parent = None):
        super(MainGUI, self).__init__(parent)

        
        self.tab1 = ProcessTab()
        self.tab1Scroll = QScrollArea()
        self.tab1Scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.tab1Scroll.setWidgetResizable(True)
        self.tab1Scroll.setWidget(self.tab1)
        self.setGeometry(0, 0, 1920, 1080)
        self.addTab(self.tab1Scroll,"Tab1")





if __name__ == '__main__':
    app = QApplication([])
    #apply_stylesheet(app, theme='dark_lightgreen.xml')
    Gui = MainGUI()
    Gui.show()
    sys.exit(app.exec())