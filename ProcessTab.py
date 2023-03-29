from PyQt5.QtCore import QDateTime, Qt, QTimer, pyqtSignal,  pyqtSlot
from PyQt5.QtWidgets import (QMainWindow, QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QScrollArea)
from qt_material import apply_stylesheet
import sys
from ProcessWidget import ProcessWidget


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
    def __init__(self, parent=None):
        super(ProcessTab, self).__init__(parent)

        self.mainLayout = QHBoxLayout()
        self.setLayout(self.mainLayout)
        
        self.widgetList = []
        self.addWidget = AddWidget()
        self.addWidget.button.clicked.connect(self.addProcess)
        self.widgetList.append(self.addWidget)
        
        self.updateLayoutToList()

    def updateLayoutToList(self):
        for i in reversed(range(self.mainLayout.count())): 
            widgetToRemove = self.mainLayout.itemAt(i).widget()
            # remove it from the layout list
            self.mainLayout.removeWidget(widgetToRemove)
            # remove it from the gui
            widgetToRemove.setParent(None)

        previousWidget = None
        for widget in self.widgetList:
            self.mainLayout.addWidget(widget,1)
            if previousWidget != None and widget != self.addWidget:
                try: previousWidget.updateImageOut.disconnect()
                except TypeError: pass
                previousWidget.updateImageOut.connect(widget.setImageIn)
                previousWidget.updateImageOut.emit(previousWidget.imageOut)

            previousWidget = widget

            
    
    @pyqtSlot()
    def addProcess(self):
        processWidget = ProcessWidget()
        #processWidget   
        #processWidget.setSizePolicy(QSizePolicy.Policy.Minimum,QSizePolicy.Policy.Preferred)
        self.widgetList.insert(len(self.widgetList)-1,processWidget)
        self.updateLayoutToList()





if __name__ == '__main__':
    app = QApplication([])
    apply_stylesheet(app, theme='dark_teal.xml')
    Gui = ProcessTab()
    Gui.show()
    sys.exit(app.exec())