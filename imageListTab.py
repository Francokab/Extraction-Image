import threading
from time import sleep
from PyQt5.QtCore import QDateTime, Qt, QTimer, pyqtSignal,  pyqtSlot
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QDoubleSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QFileDialog, QScrollArea)
from qt_material import apply_stylesheet
import sys
from function import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from QDoubleSlider import QDoubleSlider
from FullScreenImage import FullScreenImage
from math import ceil
import matplotlibUtility

target_file = "images\\dragons.png"


def clearLayout(layout):
    if layout is not None:
        while layout.count():
            child = layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
            elif child.layout() is not None:
                clearLayout(child.layout())

def NImageToIndex(NImage,x,y,xmax,ymax):
    rmax = ymax/xmax
    bestR = 10000000000000000000000000
    bestIndex = (1,NImage)
    for nrow in range(1,NImage+1):
        ncol = ceil(NImage/nrow)
        r = (nrow*y)/(x*ncol)
        if abs(rmax-r)<abs(rmax-bestR):
            bestR = r
            bestIndex = (nrow, ncol)
    return (bestIndex[0], bestIndex[1])


class imageWidget(QWidget):
    closeWidget = pyqtSignal()

    def __init__(self, image, parent=None):
        super(imageWidget, self).__init__(parent)

        self.image = image
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.canvas)

        self.bottomLayout = QHBoxLayout()
        self.checkBox = QCheckBox("Inclure")
        self.bottomLayout.addWidget(self.checkBox)
        self.removeButton = QPushButton("Suprime")
        self.widgetWantClose = False
        self.removeButton.clicked.connect(self.closeWidgetFunction)
        self.bottomLayout.addWidget(self.removeButton)

        self.mainLayout.addLayout(self.bottomLayout)
        self.setLayout(self.mainLayout)
        self.setContentsMargins(0,0,0,0)
        self.plot()

    def plot(self):
        self.figure.clear()
        # create an axis
        ax = self.figure.add_subplot(111)
        # plot data
        ax.imshow(self.image, "gray")
        ax.xaxis.set_tick_params(bottom=False, labelbottom=False)
        ax.yaxis.set_tick_params(left=False, labelleft=False)
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # refresh canvas
        self.canvas.draw()
    
    def closeWidgetFunction(self):
        self.widgetWantClose = True
        self.closeWidget.emit()

class imageList(QWidget):
    def __init__(self, parent=None):
        super(imageList, self).__init__(parent)

        self.middleLayout = QHBoxLayout()
        self.middleLayout.setContentsMargins(0,0,0,0)
        self.middleLayout.setSpacing(0)
        self.middleLayout.addStretch(1)
        self.imageWidgetList = []
        self.setLayout(self.middleLayout)

class imageListTab(QGroupBox):
    def __init__(self, parent=None):
        super(imageListTab, self).__init__(parent)
        
        self.topLayout = QHBoxLayout()
        button = QPushButton()
        self.topLayout.addWidget(button)

        self.imageList = imageList()
        self.middleScroll = QScrollArea()
        self.middleScroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.middleScroll.setWidgetResizable(True)
        self.middleScroll.setContentsMargins(0,0,0,0)
        self.middleScroll.setWidget(self.imageList)
        self.middleScroll.setFixedHeight(150)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.imageToDraw = []
        self.toolbar = NavigationToolbar(self.canvas,self)
        matplotlibUtility.pre_zoom(self.figure)                     # Prepare plot event handler
        plt.connect('motion_notify_event', matplotlibUtility.re_zoom)  # for right-click pan/zoom
        plt.connect('button_release_event', matplotlibUtility.re_zoom)  # for rectangle-select zoom

        #assemble all the layout together
        self.mainLayout = QVBoxLayout()
        self.mainLayout.addLayout(self.topLayout)
        self.mainLayout.addWidget(self.middleScroll)
        self.mainLayout.addWidget(self.toolbar)
        self.mainLayout.addWidget(self.canvas)
        self.setLayout(self.mainLayout)
        

    def plot(self):
        self.figure.clear()
        # create an axis
        NImage = len(self.imageToDraw)
        if NImage != 0:
            y = self.imageToDraw[0].shape[0]
            x = self.imageToDraw[0].shape[1]
            xmax = self.canvas.sizeHint().width()
            ymax = self.canvas.sizeHint().height()
            nrow, ncol = NImageToIndex(NImage,x,y,xmax,ymax)
            for i in range(NImage):
                ax = self.figure.add_subplot(nrow,ncol,i+1)
                # plot data
                ax.imshow(self.imageToDraw[i], "gray")
                ax.xaxis.set_tick_params(bottom=False, labelbottom=False)
                ax.yaxis.set_tick_params(left=False, labelleft=False)
                self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            
            # refresh canvas
            matplotlibUtility.pre_zoom(self.figure)
            self.canvas.draw()
        
    def putImageInFullScreen(self):
        self.fullScreen = FullScreenImage(self.imageOut)
        self.fullScreen.show()
    
    @pyqtSlot(list)
    def savedImageHandler(self,image):
        for _image in image:
            self.appendToImageList(_image)
    
    def appendToImageList(self,image):
        widget = imageWidget(image)
        widget.setFixedWidth(175)
        widget.checkBox.clicked.connect(self.updateImageToDraw)
        widget.closeWidget.connect(self.closeWidgetHandler)
        self.imageList.imageWidgetList.append(widget)
        self.imageList.middleLayout.insertWidget(len(self.imageList.imageWidgetList)-1,widget)
    
    def removeFromWidgetList(self,index):
        widget = self.imageList.imageWidgetList.pop(index)
        self.imageList.middleLayout.removeWidget(widget)
        widget.deleteLater()
        self.updateImageToDraw()


    def updateImageToDraw(self):
        self.imageToDraw = []
        for widget in self.imageList.imageWidgetList:
            if widget.checkBox.isChecked():
                self.imageToDraw.append(widget.image)
        
        self.plot()
    
    def closeWidgetHandler(self):
        i = 0
        while (len(self.imageList.imageWidgetList)>i):
            if self.imageList.imageWidgetList[i].widgetWantClose:
                self.removeFromWidgetList(i)
            else:
                i = i + 1


if __name__ == '__main__':
    app = QApplication([])
    apply_stylesheet(app, theme='dark_teal.xml')
    Gui = imageListTab()
    Gui.show()
    sys.exit(app.exec())