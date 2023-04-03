from PyQt5.QtCore import QDateTime, Qt, QTimer, pyqtSignal,  pyqtSlot
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget)
from qt_material import apply_stylesheet
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

class FullScreenImage(QMainWindow):
    def __init__(self, image, parent=None):
        super(FullScreenImage, self).__init__(parent)
        self._main = QWidget()
        self.setCentralWidget(self._main)
        layout = QVBoxLayout(self._main)

        #create the widget for the image
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.addToolBar(NavigationToolbar(self.canvas,self))
        self.image = image

        self.layout().setContentsMargins(0,0,0,0)
        self.plot()
        self.showMaximized()


    def plot(self):
        self.figure.clear()
        # create an axis
        NImage = len(self.image)
        for i in range(NImage):
            ax = self.figure.add_subplot(NImage,1,i+1)
            # plot data
            ax.imshow(self.image[i], "gray")
            ax.xaxis.set_tick_params(bottom=False, labelbottom=False)
            ax.yaxis.set_tick_params(left=False, labelleft=False)
            self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # refresh canvas
        self.canvas.draw()
        


if __name__ == '__main__':
    app = QApplication([])
    apply_stylesheet(app, theme='dark_teal.xml')
    Gui = FullScreenImage()
    Gui.show()
    sys.exit(app.exec())