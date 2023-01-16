
FUNCTION_DICT = dict()

class imgProcessingFunc:
    def __init__(self,func) -> None:
        self.func = func
        self.name = func.__name__
        self.doc = func.__doc__
        self.type = None
    
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

def imageReadingGUI(func):
    func1 = imgProcessingFunc(func)
    func1.type = "imgReading"
    FUNCTION_DICT[func1.name] = func1
    return func