from PyQt5.QtCore import QDateTime, Qt, QTimer, pyqtSignal,  pyqtSlot
from PyQt5.QtWidgets import (QMainWindow, QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QScrollArea)
from qt_material import apply_stylesheet
import sys
from ProcessWidget import ProcessWidget, clearLayout
from decoratorGUI import *
from function import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from FullScreenImage import FullScreenImage


class AddWidget(QWidget):
    def __init__(self,parent=None):
        super(AddWidget, self).__init__(parent)
        
        self.mainLayout = QVBoxLayout()
        self.mainLayout.addStretch(1)
        self.button = QPushButton("Add")
        self.mainLayout.addWidget(self.button)
        self.mainLayout.addStretch(1)
        self.setLayout(self.mainLayout)

class imgReadingWidget(QGroupBox):
    updateImageOut = pyqtSignal(list)
    saveImageOut = pyqtSignal(list)
    closeWidget = pyqtSignal()

    def __init__(self, parent=None):
        super(imgReadingWidget, self).__init__(parent)
        
        #Create the top of the top of the layer
        resetButton = QPushButton("Reset")

        removeButton = QPushButton("Fermer")
        self.widgetWantClose = False
        removeButton.clicked.connect(self._closeWidget)

        topTopLayout = QHBoxLayout()
        topTopLayout.addWidget(resetButton)
        topTopLayout.addWidget(removeButton)


        #Create the select box at the top of the widget
        self.imgComboBox = QComboBox()
        self.imgComboBox.addItems(IMAGE_DICT.keys())
        self.imgComboBox.currentTextChanged.connect(self.changeImg)
        self.imgComboBox.setToolTip("Selectionne une image")

        #Label for the combo box
        imgProcessLabel = QLabel("Image : ")
        imgProcessLabel.setToolTip("Selectionne une image")
        imgProcessLabel.setBuddy(self.imgComboBox)

        #set the layout for the combo box
        self.topLayout = QHBoxLayout()
        self.topLayout.addStretch(1)
        self.topLayout.addWidget(imgProcessLabel)
        self.topLayout.addWidget(self.imgComboBox)

        #create the widget for the image
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.imageOut = None

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

        self.imgComboBox.setCurrentText("Dragons")
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

    def changeImg(self,imgName):
        if imgName != " ---- ":
            self.imageOut = [readImageFromFile(IMAGE_DICT[imgName])]
            self.plot()

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
        
    def putImageInFullScreen(self):
        self.fullScreen = FullScreenImage(self.imageOut)
        self.fullScreen.show()
    
    def saveImage(self):
        self.setBackgroundGreen()
        self.saveImageOut.emit(self.imageOut)
        self.setBackgroundBlanc()
    
    def _closeWidget(self):
        self.widgetWantClose = False
        self.closeWidget.emit()

class ProcessTab(QWidget):
    saveImageOut = pyqtSignal(list)

    def __init__(self, parent=None):
        super(ProcessTab, self).__init__(parent)

        self.topLayout = QHBoxLayout()
        applyButton = QPushButton("Appliquer l'algorithme sélectionné")
        applyButton.clicked.connect(self.applyAlgo)
        self.topLayout.addWidget(applyButton)

        algoComboBox = QComboBox()
        algoComboBox.addItem(" ---- ")
        self.algo = None
        algoComboBox.addItems(ALGO_DICT.keys())
        algoComboBox.currentTextChanged.connect(self.algoChange)
        algoComboBox.setToolTip("Selectionne un algorithme à tester")
        self.topLayout.addWidget(algoComboBox)
        self.topLayout.addStretch(2)

        self.mainLayout = QHBoxLayout()
        self.mainMainLayout = QVBoxLayout()
        self.mainMainLayout.addLayout(self.topLayout)
        self.mainMainLayout.addLayout(self.mainLayout)
        self.setLayout(self.mainMainLayout)
        
        self.widgetList = []
        self.imgReadingWidget = imgReadingWidget()
        self.imgReadingWidget.setFixedWidth(267)
        self.imgReadingWidget.saveImageOut.connect(self.savedImageHandler)
        self.imgReadingWidget.closeWidget.connect(self.closeWidgetHandler)
        self.appendToWidgetList(self.imgReadingWidget)
        self.addWidget = AddWidget()
        self.addWidget.button.clicked.connect(self.addProcess)
        self.addWidget.setFixedWidth(267)
        self.mainLayout.addWidget(self.addWidget)
        self.mainLayout.addStretch(1)


    def appendToWidgetList(self, widget):
        if len(self.widgetList)>0:
            previousWidget = self.widgetList[-1]
        else:
            previousWidget = None
        self.widgetList.append(widget)

        if previousWidget is not None:
            previousWidget.updateImageOut.connect(widget.setImageIn)
            try: previousWidget.updateImageOut.emit(previousWidget.imageOut)
            except TypeError: pass
        self.mainLayout.insertWidget(len(self.widgetList)-1,widget)
            
    
    @pyqtSlot()
    def addProcess(self):
        processWidget = ProcessWidget()
        processWidget.setFixedWidth(267)
        processWidget.saveImageOut.connect(self.savedImageHandler)
        processWidget.closeWidget.connect(self.closeWidgetHandler)
        self.appendToWidgetList(processWidget)

    def removeFromWidgetList(self,index):
        if index>0:
            previousWidget = self.widgetList[index-1]
        else:
            previousWidget = None
        if index<len(self.widgetList)-1:
            nextWidget = self.widgetList[index+1]
        else:
            nextWidget = None
        widget = self.widgetList.pop(index)
        self.mainLayout.removeWidget(widget)
        widget.deleteLater()
        
        if previousWidget is not None and nextWidget is not None:
            previousWidget.updateImageOut.connect(nextWidget.setImageIn)
            try: previousWidget.updateImageOut.emit(previousWidget.imageOut)
            except TypeError: pass

    def deleteAllWidgetList(self):
        while (len(self.widgetList)>1):
            self.removeFromWidgetList(len(self.widgetList)-1)

    def algoChange(self,algoString):
        if algoString != " ---- ":
            self.algo = ALGO_DICT[algoString]
        else:
            self.algo = None

    def applyAlgo(self, algo):
        if self.algo is not None:
            self.deleteAllWidgetList()
            for funcString in self.algo.functionList:
                self.addProcess()
                self.widgetList[-1].imgProcessComboBox.setCurrentText(FUNCTION_DICT[funcString].displayName)

    @pyqtSlot(list)
    def savedImageHandler(self,image):
        self.saveImageOut.emit(image)
    
    def closeWidgetHandler(self):
        i = 0
        while (len(self.widgetList)>i):
            if self.widgetList[i].widgetWantClose:
                self.removeFromWidgetList(i)
            else:
                i = i + 1

if __name__ == '__main__':
    app = QApplication([])
    apply_stylesheet(app, theme='dark_teal.xml')
    Gui = ProcessTab()
    Gui.show()
    sys.exit(app.exec())