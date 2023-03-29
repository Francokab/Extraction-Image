
FUNCTION_DICT = dict()

class imgProcessingFunc:
    def __init__(self,func) -> None:
        self.func = func
        self.name = func.__name__
        self.doc = func.__doc__
        self.type = None
        self.parameters = []
    
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class parameter:
    def __init__(self,docString):
        # format in the doc
        # name, type
        # end_parameter
        docString = docString.split(", ")
        self.name = docString[0]
        self.type = docString[1]
        self.value = None
    
    def setValue(self, value):
        #print(value)
        self.value = value
        

def imageReadingGUI(func):
    func1 = imgProcessingFunc(func)
    func1.type = "imgReading"
    FUNCTION_DICT[func1.name] = func1
    return func

def noParameterGUI(func):
    func1 = imgProcessingFunc(func)
    func1.type = "noParameter"
    FUNCTION_DICT[func1.name] = func1
    return func

def parameterGUI(func):
    func1 = imgProcessingFunc(func)
    func1.type = "parameter"
    doc = func1.doc.split("\n")
    doc.pop(0)
    while(doc[0].strip(" ") != "end_parameter"):
        func1.parameters.append(parameter(doc.pop(0).strip(" ")))

    doc.pop(0)
    func1.doc = "\n".join(doc)
    FUNCTION_DICT[func1.name] = func1
    return func