from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QFileDialog)
from qt_material import apply_stylesheet
import sys
from function import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

target_file = "images\\Sunflowers_in_July.jpg"

class ProcessWidget(QGroupBox):
    def __init__(self, parent=None):
        super(ProcessWidget, self).__init__(parent)
        
        imgProcessComboBox = QComboBox()
        imgProcessComboBox.addItem(" ---- ")
        self.func = None
        imgProcessComboBox.addItems(FUNCTION_DICT.keys())
        imgProcessComboBox.activated[str].connect(self.changeImgProcess)

        imgProcessLabel = QLabel("Function : ")
        imgProcessLabel.setBuddy(imgProcessComboBox)

        self.topLayout = QHBoxLayout()
        self.topLayout.addStretch(1)
        self.topLayout.addWidget(imgProcessLabel)
        self.topLayout.addWidget(imgProcessComboBox)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.imageOut = None

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addLayout(self.topLayout)
        self.mainLayout.addWidget(self.canvas)

        self.setLayout(self.mainLayout)

    def resetMainLayout(self):
        while(self.mainLayout.itemAt(2) is not None):
            self.mainLayout.removeItem(self.mainLayout.itemAt(2))

    def changeImgProcess(self, imgProcessName):
        self.resetMainLayout()
        if imgProcessName != " ---- ":
            self.func = FUNCTION_DICT[imgProcessName]
            if self.func.type == "imgReading":
                widget = QFileDialog()
                self.mainLayout.addWidget(widget)
            
        else:
            self.func = None

    def plot(self):
        self.figure.clear()
        # create an axis
        ax = self.figure.add_subplot(111)
        # plot data
        ax.imshow(self.imageOut)
        # refresh canvas
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication([])
    apply_stylesheet(app, theme='dark_teal.xml')
    Gui = ProcessWidget()
    Gui.show()
    sys.exit(app.exec())