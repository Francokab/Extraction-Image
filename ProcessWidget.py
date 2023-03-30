import threading
from time import sleep
from PyQt5.QtCore import QDateTime, Qt, QTimer, pyqtSignal,  pyqtSlot
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QDoubleSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QFileDialog)
from qt_material import apply_stylesheet
import sys
from function import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from QDoubleSlider import QDoubleSlider

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
        
        #Create the select box at the top of the widget
        imgProcessComboBox = QComboBox()
        imgProcessComboBox.addItem(" ---- ")
        imgProcessComboBox.addItems(FUNCTION_DICT.keys())
        imgProcessComboBox.currentTextChanged.connect(self.changeImgProcess)
        imgProcessComboBox.setToolTip("Selectionne une fonction à appliquer à l'étape précedente")
        
        #create the event that check when to run the function
        self.func = None
        self.doProcessLater = threading.Event()
        self.checkProcessTimer = QTimer(self)
        self.checkProcessTimer.setInterval(500)
        self.checkProcessTimer.timeout.connect(self.doProcess)
        self.checkProcessTimer.start()

        #Label for the combo box
        imgProcessLabel = QLabel("Function : ")
        imgProcessLabel.setToolTip("Selectionne une fonction à appliquer à l'étape précedente")
        imgProcessLabel.setBuddy(imgProcessComboBox)

        #set the layout for the combo box
        self.topLayout = QHBoxLayout()
        self.topLayout.addStretch(1)
        self.topLayout.addWidget(imgProcessLabel)
        self.topLayout.addWidget(imgProcessComboBox)

        #create the widget for the image
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.imageOut = None
        self.imageIn = None

        #create the layout for the parameters widgets
        self.bottomLayout = QGridLayout()
        self.resetBottomLayout()

        #assemble all the layout together
        self.mainLayout = QVBoxLayout()
        self.mainLayout.addLayout(self.topLayout)
        self.mainLayout.addWidget(self.canvas)
        self.mainLayout.addLayout(self.bottomLayout)
        self.mainLayout.addStretch(1)

        self.setLayout(self.mainLayout)
        self.layout().setContentsMargins(0,0,0,0)

    def resetBottomLayout(self):
        clearLayout(self.bottomLayout)
        self.bottomLayout.addWidget(QLabel("Info"),0,1)
        self.bottomLayout.setColumnStretch(0,1)
        self.bottomLayout.setColumnStretch(1,1)


    def changeImgProcess(self, imgProcessName):
        self.resetBottomLayout()
        if imgProcessName != " ---- ":
            self.func = FUNCTION_DICT[imgProcessName]
            self.topLayout.itemAt(2).widget().setToolTip(self.func.name)
            if self.func.type == "imgReading":
                pass
            elif self.func.type == "parameter":
                index = 0
                for param in self.func.parameters:
                    if param.type == "image":
                        haveWidget = False
                    elif param.type == "int":
                        widget = QSpinBox()
                        widget.valueChanged.connect(param.setValue)
                        widget.setValue(param.default)
                        widget.setMinimum(param.min)
                        widget.setMaximum(param.max)
                        widget.valueChanged.connect(self.doProcessLater.set)
                        haveWidget = True

                    elif param.type == "float":
                        widget = QDoubleSpinBox()
                        widget.setSingleStep(0.1)
                        widget.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
                        widget.valueChanged.connect(param.setValue)
                        widget.setValue(param.default)
                        widget.setMinimum(param.min)
                        widget.setMaximum(param.max)
                        widget.valueChanged.connect(self.doProcessLater.set)
                        haveWidget = True
                    
                    elif param.type == "slider":
                        widget = QDoubleSlider(3,Qt.Horizontal)
                        widget.doubleValueChanged.connect(param.setValue)
                        widget.doubleValueChanged.connect(print)
                        widget.setValue(param.default)
                        widget.setMinimum(param.min)
                        widget.setMaximum(param.max)
                        widget.doubleValueChanged.connect(self.doProcessLater.set)
                        haveWidget = True

                    elif param.type == "list":
                        widget = QComboBox()
                        widget.addItems(param.list)
                        widget.currentTextChanged.connect(param.setValue)
                        widget.setCurrentText(param.default)
                        widget.currentTextChanged.connect(self.doProcessLater.set)
                        haveWidget = True
                    else:
                        haveWidget = False

                    if haveWidget:
                        layout = QHBoxLayout()
                        label = QLabel(param.displayName)
                        label.setToolTip(param.description)
                        layout.addWidget(label)
                        layout.addWidget(widget)
                        self.bottomLayout.addLayout(layout,1+index//2,index%2)
                        index += 1

            
        else:
            self.func = None
            self.topLayout.itemAt(2).widget().setToolTip("Selectionne une fonction à appliquer à l'étape précedente")
        self.doProcessLater.set()
        

    def doProcess(self):
        if self.doProcessLater.is_set():
            if self.func == None:
                self.imageOut = self.imageIn
            elif self.func.type == "imgReading":
                self.imageOut = [self.func(target_file)]
            elif self.func.type == "parameter":
                imageParamList = [param for param in self.func.parameters if param.type == "image"]
                for i, param in enumerate(imageParamList):
                    param.setValue(self.imageIn[i])
                inputDict = {param.name:param.value for param in self.func.parameters}
                self.imageOut = self.func(**inputDict)
                if type(self.imageOut) == tuple:
                    self.imageOut = list(self.imageOut)
                else:
                    self.imageOut = [self.imageOut]
            
            self.plot()
            self.doProcessLater.clear()
                  
    def plot(self):
        self.figure.clear()
        # create an axis
        NImage = len(self.imageOut)
        for i in range(NImage):
            ax = self.figure.add_subplot(NImage,1,i+1)
            # plot data
            ax.imshow(self.imageOut[0], "gray")
            ax.xaxis.set_tick_params(bottom=False, labelbottom=False)
            ax.yaxis.set_tick_params(left=False, labelleft=False)
            self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # refresh canvas
        self.canvas.draw()
        self.updateImageOut.emit(self.imageOut)

    @pyqtSlot(list)
    def setImageIn(self,imageIn):
        self.imageIn = imageIn
        self.doProcessLater.set()
        


if __name__ == '__main__':
    app = QApplication([])
    apply_stylesheet(app, theme='dark_teal.xml')
    Gui = ProcessWidget()
    Gui.show()
    sys.exit(app.exec())