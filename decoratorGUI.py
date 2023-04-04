from copy import deepcopy
FUNCTION_DICT = dict()
SECONDARY_FUNCTION_DICT = dict()
ALGO_DICT = dict()

class imgProcessingFunc:
    def __init__(self,func) -> None:
        self.func = func
        self.name = func.__name__
        self.displayName = self.name
        self.doc = func.__doc__
        self.type = None
        self.parameters = []
    
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class parameter:
    def __init__(self,docString):
        # format in the doc
        # name; display_name; description; type; default; [list, list] or [min,max] or secondary function
        # for secondary function
        # [name, input1:input2:..., output1:output2:...]
        # end_parameter
        docString = docString.split("; ")
        self.name = docString[0]
        self.displayName = docString[1]
        self.description = docString[2]
        self.type = docString[3]
        self.default = None
        self.widget = None
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
            elif self.type in ["special_bool"]:
                self.setSecondaryFunction(docString[5])

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
        elif self.type in ["special_bool"]:
            self.default = bool(value)

    def setValue(self, value):
        if self.type == "image":
            self.value = value
        elif self.type == "int":
            self.value = int(value)
        elif self.type in ["float", "slider"]:
            self.value = float(value)
        elif self.type == "list":
            self.value = value
        elif self.type in ["special_bool"]:
            self.default = bool(value)
    
    def setList(self, docString):
        self.list = docString.strip("[]").split(', ') 

    def setSecondaryFunction(self,docString):
        docString = docString.strip("[]").split(', ')
        self.secondaryFunction = docString[0]
        self.input = docString[1].split(":")
        self.output = docString[2].split(":")

class algo:
    def __init__(self, name, displayName, description, functionList):
        self.name = name
        self.displayName = displayName
        self.description = description
        self.functionList = functionList

def imageReadingGUI(func):
    func1 = imgProcessingFunc(func)
    func1.type = "imgReading"
    FUNCTION_DICT[func1.name] = func1
    return func

def secondaryFunction(func):
    func1 = imgProcessingFunc(func)
    func1.type = "secondaryFunction"
    SECONDARY_FUNCTION_DICT[func1.name] = func1
    return func

def parameterGUI(func):
    func1 = imgProcessingFunc(func)
    func1.type = "parameter"
    doc = func1.doc.split("\n")
    func1.displayName = doc.pop(0)
    while(doc[0].strip(" ") != "end_parameter"):
        func1.parameters.append(parameter(doc.pop(0).strip(" ")))

    doc.pop(0)
    func1.doc = "\n".join(doc)
    FUNCTION_DICT[func1.name] = func1
    return func