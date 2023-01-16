from re import L
from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QMainWindow, QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget)
from qt_material import apply_stylesheet
import sys

class ImagesGUI(QMainWindow):
    def __init__(self, parent=None):
        super(ImagesGUI, self).__init__(parent)

        self.setWindowTitle("Extraction Image")




if __name__ == '__main__':
    app = QApplication([])
    apply_stylesheet(app, theme='dark_teal.xml')
    Gui = ImagesGUI()
    Gui.show()
    sys.exit(app.exec())