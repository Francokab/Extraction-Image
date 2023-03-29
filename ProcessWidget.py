from time import sleep
from PyQt5.QtCore import QDateTime, Qt, QTimer, pyqtSignal,  pyqtSlot
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

target_file = "images\\dragons.png"

def clearLayout(layout):
    if layout is not None:
        while layout.count():
            child = layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
            elif child.layout() is not None:
                clearLayout(child.layout())
class ProcessWidget(QGroupBox):
    updateImageOut = pyqtSignal(list)

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
        self.imageIn = None

        self.bottomLayout = QGridLayout()
        self.resetBottomLayout()

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addLayout(self.topLayout)
        self.mainLayout.addWidget(self.canvas)
        self.mainLayout.addLayout(self.bottomLayout)
        self.mainLayout.addStretch(1)

        self.setLayout(self.mainLayout)

    def resetBottomLayout(self):
        clearLayout(self.bottomLayout)
        self.bottomLayout.addWidget(QLabel("Info"),0,1)
        self.bottomLayout.setColumnStretch(0,1)
        self.bottomLayout.setColumnStretch(1,1)


    def changeImgProcess(self, imgProcessName):
        self.resetBottomLayout()
        if imgProcessName != " ---- ":
            self.func = FUNCTION_DICT[imgProcessName]
            if self.func.type == "imgReading":
                pass
            elif self.func.type == "noParameter":
                pass
            elif self.func.type == "parameter":
                index = 0
                for param in self.func.parameters:
                    if param.type == "image":
                        pass
                    elif param.type == "int":
                        spinBox = QSpinBox()
                        spinBox.valueChanged.connect(param.setValue)
                        spinBox.setValue(1)
                        layout = QHBoxLayout()
                        layout.addWidget(QLabel(param.name))
                        layout.addWidget(spinBox)
                        self.bottomLayout.addLayout(layout,1+index//2,index%2)
                        index += 1

            
        else:
            self.func = None
        print("new")
        self.doProcess()
        
    def doProcess(self):
        if self.func == None:
            self.imageOut = self.imageIn
        elif self.func.type == "imgReading":
            self.imageOut = [self.func(target_file)]
        elif self.func.type == "noParameter":
            self.imageOut = [self.func(self.imageIn[0])]
        
        self.plot()

    def plot(self):
        self.figure.clear()
        # create an axis
        ax = self.figure.add_subplot(111)
        # plot data
        ax.imshow(self.imageOut[0], "gray")
        # refresh canvas
        self.canvas.draw()
        self.updateImageOut.emit(self.imageOut)

    @pyqtSlot(list)
    def setImageIn(self,imageIn):
        self.imageIn = imageIn
        self.doProcess()


if __name__ == '__main__':
    app = QApplication([])
    apply_stylesheet(app, theme='dark_teal.xml')
    Gui = ProcessWidget()
    Gui.show()
    sys.exit(app.exec())