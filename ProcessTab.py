from PyQt5.QtCore import QDateTime, Qt, QTimer, pyqtSignal,  pyqtSlot
from PyQt5.QtWidgets import (QMainWindow, QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QScrollArea)
from qt_material import apply_stylesheet
import sys
from ProcessWidget import ProcessWidget
from decoratorGUI import *


class AddWidget(QWidget):
    def __init__(self,parent=None):
        super(AddWidget, self).__init__(parent)
        
        self.mainLayout = QVBoxLayout()
        self.mainLayout.addStretch(1)
        self.button = QPushButton("Add")
        self.mainLayout.addWidget(self.button)
        self.mainLayout.addStretch(1)
        self.setLayout(self.mainLayout)

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
        while (len(self.widgetList)>0):
            self.removeFromWidgetList(len(self.widgetList)-1)

    def algoChange(self,algoString):
        if algoString != " ---- ":
            self.algo = ALGO_DICT[algoString]
        else:
            self.algo = None

    def applyAlgo(self, algo):
        if self.algo is not None:
            self.deleteAllWidgetList()
            print(FUNCTION_DICT)
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