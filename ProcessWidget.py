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
from FullScreenImage import FullScreenImage

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
    saveImageOut = pyqtSignal(list)
    closeWidget = pyqtSignal()

    def __init__(self, parent=None):
        super(ProcessWidget, self).__init__(parent)
        
        #Create the top of the top of the layer
        resetButton = QPushButton("Reset")
        resetButton.clicked.connect(lambda:self.changeImgProcess(self.imgProcessName))

        removeButton = QPushButton("Fermer")
        self.widgetWantClose = False
        removeButton.clicked.connect(self._closeWidget)

        topTopLayout = QHBoxLayout()
        topTopLayout.addWidget(resetButton)
        topTopLayout.addWidget(removeButton)


        #Create the select box at the top of the widget
        self.imgProcessComboBox = QComboBox()
        self.imgProcessComboBox.addItem(" ---- ")
        self.imgProcessName = " ---- "
        self.imgProcessComboBox.addItems([FUNCTION_DICT[key].displayName for key in FUNCTION_DICT.keys()])
        self.imgProcessComboBox.currentTextChanged.connect(self.changeImgProcess)
        self.imgProcessComboBox.setToolTip("Selectionne une fonction à appliquer à l'étape précedente")
        
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
        imgProcessLabel.setBuddy(self.imgProcessComboBox)

        #set the layout for the combo box
        self.topLayout = QHBoxLayout()
        self.topLayout.addStretch(1)
        self.topLayout.addWidget(imgProcessLabel)
        self.topLayout.addWidget(self.imgProcessComboBox)

        #create the widget for the image
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.imageOut = None
        self.imageIn = None

        #create the layout for the parameters widgets
        self.bottomLayout = QGridLayout()
        self.resetBottomLayout()
        self.docInfo = QLabel()
        self.docInfo.setWordWrap(True)

        #assemble all the layout together
        self.mainLayout = QVBoxLayout()
        self.mainLayout.addLayout(topTopLayout)
        self.mainLayout.addLayout(self.topLayout)
        self.mainLayout.addWidget(self.canvas)
        self.mainLayout.addLayout(self.bottomLayout)
        self.mainLayout.addWidget(self.docInfo)
        self.mainLayout.addStretch(1)

        self.setLayout(self.mainLayout)
        self.layout().setContentsMargins(0,0,0,0)

    def resetBottomLayout(self):
        clearLayout(self.bottomLayout)
        fullScreenButton = QPushButton("Full Screen")
        fullScreenButton.clicked.connect(self.putImageInFullScreen)
        self.bottomLayout.addWidget(fullScreenButton,0,0)
        saveImageButton = QPushButton("Sauvergarder l'image")
        saveImageButton.clicked.connect(self.saveImage)
        self.bottomLayout.addWidget(saveImageButton,0,1)
        self.bottomLayout.setColumnStretch(0,1)
        self.bottomLayout.setColumnStretch(1,1)

    def changeImgProcess(self, imgProcessName):
        self.imgProcessName = imgProcessName
        self.resetBottomLayout()
        if imgProcessName != " ---- ":
            self.func = deepcopy(*[FUNCTION_DICT[key] for key in FUNCTION_DICT if FUNCTION_DICT[key].displayName == imgProcessName])
            self.topLayout.itemAt(2).widget().setToolTip(self.func.name)
            if self.func.type == "imgReading":
                pass
            elif self.func.type == "parameter":
                self.docInfo.setText(self.func.doc)
                index = 0
                imageIndex = 0
                for param in self.func.parameters:
                    if param.type == "image":
                        try :param.setValue(self.imageIn[imageIndex])
                        except TypeError: param.setValue(None)
                        haveWidget = False
                    elif param.type == "int":
                        widget = QSpinBox()
                        param.widget = widget
                        widget.valueChanged.connect(param.setValue)
                        widget.setMinimum(param.min)
                        widget.setMaximum(param.max)
                        widget.setValue(param.default)
                        widget.valueChanged.connect(self.doProcessLater.set)
                        haveWidget = True

                    elif param.type == "float":
                        widget = QDoubleSpinBox()
                        param.widget = widget
                        widget.setSingleStep(0.1)
                        widget.setDecimals(3)
                        widget.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
                        widget.valueChanged.connect(param.setValue)
                        widget.setMinimum(param.min)
                        widget.setMaximum(param.max)
                        widget.setValue(param.default)
                        widget.valueChanged.connect(self.doProcessLater.set)
                        haveWidget = True
                    
                    elif param.type == "slider":
                        widget = QDoubleSlider(3,Qt.Horizontal)
                        param.widget = widget
                        widget.doubleValueChanged.connect(param.setValue)
                        widget.setMinimum(param.min)
                        widget.setMaximum(param.max)
                        widget.setValue(param.default)
                        widget.doubleValueChanged.connect(self.doProcessLater.set)
                        haveWidget = True

                    elif param.type == "list":
                        widget = QComboBox()
                        param.widget = widget
                        widget.addItems(param.list)
                        widget.currentTextChanged.connect(param.setValue)
                        widget.setCurrentText(param.default)
                        widget.currentTextChanged.connect(self.doProcessLater.set)
                        haveWidget = True

                    elif param.type == "bool":
                        widget = QCheckBox()
                        param.widget = widget
                        widget.stateChanged.connect(param.setValue)
                        widget.setChecked(param.default)
                        widget.stateChanged.connect(self.doProcessLater.set)
                        haveWidget = True

                    elif param.type == "special_bool":
                        widget = QCheckBox()
                        param.widget = widget
                        widget.stateChanged.connect(param.setValue)
                        param.input = [_param for _stringParam in param.input for _param in self.func.parameters if _param.name == _stringParam]
                        param.secondaryFunction = deepcopy(SECONDARY_FUNCTION_DICT[param.secondaryFunction])
                        param.output = [_param for _stringParam in param.output for _param in self.func.parameters if _param.name == _stringParam]

                        x = lambda: param.secondaryFunction(*[_param.value for _param in param.input])
                        y = lambda *_output: [_param.widget.setValue(output) for _param, output in zip(param.output,_output)]
                        param.funcCall = lambda *_: y(*x())
                        
                        
                        widget.stateChanged.connect(param.funcCall)
                        [widget.stateChanged.connect(lambda _bool, _param = _param: _param.widget.setEnabled(not _bool)) for _param in param.output]
                        widget.setChecked(param.default)
                        widget.stateChanged.connect(self.doProcessLater.set)
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
            self.docInfo.setText("")
            self.topLayout.itemAt(2).widget().setToolTip("Selectionne une fonction à appliquer à l'étape précedente")
        self.doProcessLater.set()
        
    def doProcess(self):
        if self.doProcessLater.is_set():
            try:
                self.setBackgroundGreen()
                if self.func == None:
                    self.imageOut = self.imageIn
                elif self.func.type == "imgReading":
                    self.imageOut = [self.func(target_file)]
                elif self.func.type == "parameter":
                    imageParamList = [param for param in self.func.parameters if param.type == "image"]
                    for i, param in enumerate(imageParamList):
                        param.setValue(self.imageIn[i])
                    
                    specialBoolList = [param for param in self.func.parameters if param.type == "special_bool"]
                    for specialBool in specialBoolList:
                        if specialBool.widget.isChecked():
                            specialBool.funcCall()

                    inputDict = {param.name:param.value for param in self.func.parameters if param.type not in ["special_bool"]}
                    self.imageOut = self.func(**inputDict)
                    if type(self.imageOut) == tuple:
                        self.imageOut = list(self.imageOut)
                    else:
                        self.imageOut = [self.imageOut]
            
                self.plot()
                self.doProcessLater.clear()
            except TypeError: pass
            self.setBackgroundBlanc()

    def setBackgroundGreen(self):
        #self.setStyleSheet('background-color: limegreen')
        self.figure.patch.set_facecolor('xkcd:mint green')
        self.canvas.draw()
        self.repaint()

    def setBackgroundBlanc(self):
        #self.setStyleSheet('background-color: white')
        self.figure.patch.set_facecolor('white')
        self.canvas.draw()
        self.repaint()

    def plot(self):
        self.figure.clear()
        # create an axis
        NImage = len(self.imageOut)
        for i in range(NImage):
            ax = self.figure.add_subplot(NImage,1,i+1)
            # plot data
            ax.imshow(self.imageOut[i], "gray")
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
        
    def putImageInFullScreen(self):
        self.fullScreen = FullScreenImage(self.imageOut)
        self.fullScreen.show()
    
    def saveImage(self):
        self.setBackgroundGreen()
        self.saveImageOut.emit(self.imageOut)
        self.setBackgroundBlanc()
    
    def _closeWidget(self):
        self.widgetWantClose = True
        self.closeWidget.emit()

if __name__ == '__main__':
    app = QApplication([])
    apply_stylesheet(app, theme='dark_teal.xml')
    Gui = ProcessWidget()
    Gui.show()
    sys.exit(app.exec())