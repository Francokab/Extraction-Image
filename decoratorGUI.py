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
        # name; display_name; description; type; default; [list, list] ou [min,max]
        # end_parameter
        docString = docString.split("; ")
        self.name = docString[0]
        self.displayName = docString[1]
        self.description = docString[2]
        self.type = docString[3]
        self.default = None
        if (len(docString)>4):
            self.setDefault(docString[4])
            if self.type == "list":
                self.list = []
                self.setList(docString[5])
            elif self.type in ["int", "float", "slider"]:
                self.list = []
                self.setList(docString[5])
                if self.type == "int":
                    self.min = int(self.list[0])
                    self.max = int(self.list[1])
                elif self.type in ["float", "slider"]:
                    self.min = float(self.list[0])
                    self.max = float(self.list[1])
        self.value = None
        self.setValue(self.default)

    
    def setDefault(self, value):
        if self.type == "image":
            pass
        elif self.type == "int":
            self.default = int(value)
        elif self.type in ["float", "slider"]:
            self.default = float(value)
        elif self.type == "list":
            self.default = value

    def setValue(self, value):
        if self.type == "image":
            self.value = value
        elif self.type == "int":
            self.value = int(value)
        elif self.type in ["float", "slider"]:
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

def parameterGUI(func):
    func1 = imgProcessingFunc(func)
    func1.type = "parameter"
    doc = func1.doc.split("\n")
    func1.name = doc.pop(0)
    while(doc[0].strip(" ") != "end_parameter"):
        func1.parameters.append(parameter(doc.pop(0).strip(" ")))

    doc.pop(0)
    func1.doc = "\n".join(doc)
    FUNCTION_DICT[func1.name] = func1
    return func