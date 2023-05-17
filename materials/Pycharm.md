# Pycharm Tutorial
## Setup & Basics 
An interpreter is the program that interprete your code  

1. Find= Ctrl+F 
2. See changes= Alt+Shift+C
3. Duplicate Lines:  Ctrl+D 
4. To do list: #Comments shows where is the file and where should we strore the files 
5. ALT: only select all  rows vertically 
6. Surrounded with: Ctrl+Alt+T -- save time to give expressions
7. How to understand if__name__='main'
：__name__ 是当前模块名，当模块被直接运行时模块名为 __main__ 。当模块被直接运行时，代码将被运行，当模块是被导入时，代码不被运行。
由于每个Python模块（Python文件）都包含内置的变量__name__，当运行模块被执行的时候，__name__等于文件名（包含了后缀.py）。如果import到其他模块中，则__name__等于模块名称（不包含后缀.py）。而“__main__”等于当前执行文件的名称（包含了后缀.py）。所以当模块被直接执行时，__name__ == '__main__'结果为真；而当模块被import到其他模块中时，__name__ == '__main__'结果为假，就是不调用对应的方法。



## Debugging 
Debugger, shows the variables and shows the different variables in the scope;
- Add a break 
- Stop debug: 右击可以出现Debug  
- step in/step out/step over
