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
        # name; display_name; type; default; [list, list]
        # end_parameter
        docString = docString.split("; ")
        self.name = docString[0]
        self.displayName = docString[1]
        self.type = docString[2]
        self.default = None
        if (len(docString)>3):
            self.setDefault(docString[3])
        if self.type == "list":
            self.list = []
            self.setList(docString[4])
        self.value = None
        self.setValue(self.default)

    
    def setDefault(self, value):
        if self.type == "image":
            pass
        elif self.type == "int":
            self.default = int(value)
        elif self.type == "float":
            self.default = float(value)
        elif self.type == "list":
            self.default = value

    def setValue(self, value):
        if self.type == "image":
            self.value = value
        elif self.type == "int":
            self.value = int(value)
        elif self.type == "float":
            self.value = float(value)
        elif self.type == "list":
            self.value = value
    
    def setList(self, docString):
        self.list = docString.strip("[]").split(', ')

        

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