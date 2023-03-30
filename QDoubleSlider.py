from PyQt5.QtCore import pyqtSignal,  pyqtSlot
from PyQt5.QtWidgets import QSlider

class QDoubleSlider(QSlider):

    # create our our signal that we can connect to if necessary
    doubleValueChanged = pyqtSignal(float)

    def __init__(self, decimals=3, *args, **kargs):
        super(QDoubleSlider, self).__init__( *args, **kargs)
        self._multi = 10 ** decimals

        self.valueChanged.connect(self.emitDoubleValueChanged)

    def emitDoubleValueChanged(self):
        value = float(super(QDoubleSlider, self).value())/self._multi
        self.doubleValueChanged.emit(value)

    def value(self):
        return float(super(QDoubleSlider, self).value()) / self._multi

    def setMinimum(self, value):
        return super(QDoubleSlider, self).setMinimum(int(value * self._multi))

    def setMaximum(self, value):
        return super(QDoubleSlider, self).setMaximum(int(value * self._multi))

    def setSingleStep(self, value):
        return super(QDoubleSlider, self).setSingleStep(int(value * self._multi))

    def singleStep(self):
        return float(super(QDoubleSlider, self).singleStep()) / self._multi

    def setValue(self, value):
        super(QDoubleSlider, self).setValue(int(value * self._multi))