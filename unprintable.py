import math
import random
import sys
import string

class ValyrioError(Exception):

    def __init__(self,msg=None,char=0):
        self.msg = msg
        self.char = char
        if msg is None:
            super(ValyrioError,self).__init__('Error found in code')
        else:
            super(ValyrioError,self).__init__('Char {0}: {1}'.format(char,msg))

    def __repr__(self):
        return 'Char {0} produced {1}'.format(self.char,self.msg)

class InterpreterError(Exception):

    def __init__(self):
        msg = 'Error found in interpretation.'
        super(InterpreterError,self).__init__(msg)

class Stack:

    ''' Create a class equivilent to a list '''
  
    def __init__(self):
        
        self.stack = []
        self.value = None
        self.delim = 32

        self.pSuppress = True
        self.apply = False
        self.rangeStep = False
        self.suppress = False

    def __repr__(self):
        if not any(self.stack):
            return '()'
        return str(tuple(self.stack))

    def __iter__(self):
        return iter(self.stack)

    def __len__(self):
        return len(self.stack)

    def saveVal(self):
        ''' Create a variable from the first value in the stack'''
        if self.apply:
            self.value = list(self.stack)
        self.value = self.stack[0]

    def getVal(self):
        ''' Push the created variable '''
        if self.value is not None:
            self.push(self.value)

    @staticmethod
    def _isPrime(x):
        ''' Create a static function to check for primes '''
        if x in [0,1]:
            return False
        for i in range(2,x):
            if x % i == 0:
                return False
        return True

    ''' Input Commands '''

    def input(self,value):
        ''' Push an inputted value '''
        if type(value) == int:
            self.push(value)
        else:
            if '"' in value:
                value = ''.join(value)
            self.push(value)

    def inputList(self,iterable):
        self.push(*list(iterable))

    def inputRoman(self,input_):
        ''' Take roman numerals, convert to numbers and push '''
        total = 0
        lookup = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
        slookup = {k:v for k,v in zip(lookup.keys(),map(str,lookup.values()))}
        s = ''
        comp = ''
        l = False
        
        for i in input_:
            s += slookup[i]+'|'

        s = list(filter(None,s.split('|')))
        
        for i in range(len(s)-1):
            if l:
                l = False
                continue
            
            if int(s[i]) < int(s[i+1]):
                x = str(int(s[i+1]) - int(s[i]))
                l = True
            else:
                x = s[i]
                
            comp += x+'|'

        comp = list(filter(None, comp.split('|')))

        for c in comp:
            total += int(c)

        self.push(total)
                
    def moveInput(self,mapping):
        ''' Take a map and move the items in the stack according to the values '''
        cp = self.stack.copy()
        self.stack.clear()
        empty = []
        mapping = dict(mapping)

        for i in range(len(cp)):
            if i in mapping.keys():
                empty.append(cp[mapping[i]])
            else:
                empty.append(cp[i])
            
        self.push(*reversed(empty))

    ''' Stack Commands '''

    def pop(self,index=0):
        ''' Remove a value from the stack and return it '''
        return self.stack.pop(index)
            
    def push(self,*values):
        ''' Push the values to the stack.
            If the value is a string, push the ASCII numbers'''
      
        if self.stack == [0]*100:
            self.stack = []
                
        for i in values:
            if isinstance(i,str):
                for c in i:
                    self.stack.append(ord(c))
            elif isinstance(i,int):
                self.stack.append(int(i))
            elif isinstance(i,float):
                self.stack.append(int(i))
            else:
                raise ValyrioError

    ''' Nilad Commands '''

    def abs(self):
        ''' Push the absolute values '''
        if self.apply:
            for i in range(len(self)):
                self.stack[i] = abs(self.stack[i])
        else:
            self.stack[-1] = abs(self.stack[-1])

    def add(self):
        if self.apply:
            x = self.pop()
            for i in range(len(self)):
                self.stack[i] = self.stack[i] + x
        else:
            self.push(self.pop() + self.pop())

    def all(self):
        self.push(all(self.stack))

    def and_(self):
        self.push(self.pop() and self.pop())

    def any(self):
        self.push(any(self.stack))

    def applyAll(self):
        self.apply = True

    def ascall(self):
        for i in map(chr,self.stack):
            print(i,end='')
        if self.stack:
            print(end=chr(self.delim))

    def asciiset(self):
        copy = self.stack.copy()
        self.stack.clear()
        for i in copy:
            if i >= 32 and i <= 126:
                self.push(i)

    def ascone(self):
        print(chr(self.stack[-1]),end=chr(self.delim))

    def boolConvert(self):
        if self.applyAll:
            self.stack = list(map(bool, self.stack))
        else:
            self.stack[-1] = bool(self.stack)

    def concatenate(self):
        s = []
        for i in self.stack:
            s.append(str(i))
        s = ''.join(s)
        stack.push(s)

    def copy(self):
        self.push(*self.stack.copy())

    def count(self):
        return self.stack.count(self.pop())

    def cubeNumbers(self):
        self.push(*[i**3 for i in range(1,self.pop()+1)])

    def dupeLen(self):
        x = self.stack.copy()
        for i in range(len(self)):
            self.push(*x)

    def factorial(self):
        self.push(math.factorial(self.pop()))

    def fibonacci(self):
        iters = self.pop()
        a,b = 1,1
        stack.push(a,b)
        for i in range(iters-2):
            a,b = b,a+b
            self.push(b)

    def floordiv(self):
        if self.apply:
            x = self.pop()
            for i in range(len(self.stack)):
                self.stack[i] //= x
        else:
            self.push(self.pop() // self.pop())

    def ge(self):
        self.push(self.stack.pop() >= self.stack.pop())

    def gt(self):
        self.push(self.stack.pop() > self.stack.pop())

    def half(self):
        c = self.stack.copy()
        self.stack.clear()
        c = c[len(c)//2:]
        self.push(*c)

    def helloWorld(self):
        self.push(''.join('Hello, World!'))

    def is_(self):
        if self.apply:
            for i in range(len(self.stack)):
                if self.stack[i-1] != self.stack[i]:
                    self.push(0)
                    return
            self.push(1)
        else:
            self.push(self.stack.pop() == self.stack.pop())

    def isNot(self):
        self.push(self.stack.pop() != self.stack.pop())

    def isPrime(self):
        self.push(Stack._isPrime(self.stack[-1]))
      
    def isSorted(self):
        self.push(self.stack in [sorted(self.stack),sorted(self.stack,reverse=True)])

    def le(self):
        self.push(self.stack.pop() <= self.stack.pop())

    def leftShift(self):
        self.push(self.pop())

    def len(self):
        self.push(len(self))

    def log(self):
        self.push(int(math.log(self.pop(),self.pop())))

    def lt(self):
        self.push(self.stack.pop() < self.stack.pop())

    def max(self):
        self.push(max(self.stack))

    def min(self):
        self.push(min(self.stack))

    def mod(self):
        if self.apply:
            x = self.pop()
            for i in range(len(self.stack)):
                self.stack[i] %= x
        else:
            self.push(self.pop() % self.pop())

    def mul(self):
        if self.apply:
            x = self.pop()
            for i in range(len(self.stack)):
                self.stack[i] *= x
        else:
            self.push(self.pop() * self.pop())

    def not_(self):
        self.stack.push(int(not self.stack.pop()))

    def numall(self):
        for i in self.stack:
            print(i,end=' ')
        if self.stack:
            print(end=chr(self.delim))

    def numone(self):
        print(self.stack.pop(), end=chr(self.delim))

    def or_(self):
        self.push(self.stack.pop() or self.stack.pop())

    def palindrome(self):
        self.push(self.stack == self.stack[::-1])

    def pow(self):
        if self.apply:
            x = self.pop()
            for i in range(len(self)):
                self.stack[i] **= x
        else:
            self.push(self.pop() ** self.pop())

    def printSuppress(self):
        self.pSuppress = not self.pSuppress

    def push10(self):
        self.push(10)

    def push50(self):
        self.push(50)

    def push100(self):
        self.push(100)

    def push500(self):
        self.push(500)

    def pushIndex(self):
        self.push(self.stack[self.pop()])

    def pushThou(self):
        self.push(1000)

    def pushTop(self):
        self.push(self.stack[-1])

    def quit(self):
        quit_(self.pop())

    def randomInt(self):
        self.push(random.randint(self.pop(),self.pop()))

    def range(self):
        x = self.pop()
        if x < 0:
            step = -[1 if not self.rangeStep else self.pop()][0]
            inc = -1
        else:
            step = [1 if not self.rangeStep else self.pop()][0]
            inc = 1
            
        self.push(*range(0,x+inc,step))

    def removeEven(self):
        scopy = [x for x in self.stack.copy() if x % 2 == 0]
        self.stack.clear()
        self.push(*scopy)

    def removeOdd(self):
        scopy = [x for x in self.stack.copy() if x % 2 == 1]
        self.stack.clear()
        self.push(*scopy)

    def reverse(self):
        copy = self.stack.copy()
        self.stack.clear()
        copy = copy
        self.push(*copy)

    def rightShift(self):
        self.stack.insert(0,self.pop(-1))

    def root(self):
        n = self.stack.pop()
        x = self.stack.pop()
        self.push(n**(1/x))

    def round(self):
        
        if self.apply:
            
            n = self.pop()
                
            for i in range(len(self)):
                self.stack[i] = round(self.stack[i],-abs(n))
                
        else:
            
            self.stack[0] = round(self.pop(),-abs(self.pop()))
        
    def set(self):
        copy = self.stack.copy()
        self.stack.clear()
        copy = set(copy)
        self.push(*copy)

    def setDelim(self):
        self.delim = self.pop()

    def squareNumbers(self):
        self.push(*[i*i for i in range(1,self.pop()+1)])

    def sqrt(self):
        self.push(math.sqrt(self.pop()))

    def sub(self):
        if self.apply:
            x = self.pop()
            for i in range(len(self)):
                self.stack[i] -= x
        else:
            self.push(self.pop() - self.pop())

    def sum(self):
        self.push(sum(self.stack))
        
    def toggleDefaultOut(self):
        self.suppress = not self.suppress
        if not self.pSuppress:
            print('ł',end=chr(self.delim))

    def toggleRange(self):
        self.rangeStep = not self.rangeStep

    def triangleNumbers(self):
        self.push(*[sum(range(1,i+1)) for i in range(1,self.pop()+1)])

    def xPowNumbers(self):
        x = self.pop()+1
        self.push(*[i**self.pop() for i in range(1,x)])
        
stack = Stack()
for i in range(100):
    stack.push(0)

def quit_(status=1):
    if status == 1:
        quit()

SPECIALS = ['[',']','{','}','(',')','¿','?',"'",'\\','E','N','y']

COMMANDS = {

    # "All" Commands

    'b':stack.printSuppress,
    'Z':stack.applyAll,
    'Y':stack.toggleRange,
    
    # Keywords
 
    '=':stack.is_,
    '|':stack.or_,
    '!':stack.not_,
    '&':stack.and_,

    'a':stack.any,
    'A':stack.all,
    '.':stack.isNot,

    # Math Operands

    '<':stack.lt,
    '>':stack.gt,
    ';':stack.le,
    ':':stack.ge,
    '-':stack.sub,
    '+':stack.add,
    '*':stack.mul,
    '%':stack.mod,
    '^':stack.pow,
    '/':stack.floordiv,

    '$':stack.abs,
    '_':stack.sum,
    '£':stack.sqrt,
    'x':stack.root,

    'l':stack.log,
    'r':stack.range,
    'p':stack.isPrime,
    'f':stack.factorial,

    # Number Commands

    '¥':stack.round,
    
    'R':stack.randomInt,
    'X':stack.push10,
    'L':stack.push50,
    'C':stack.push100,
    'D':stack.push500,
    'T':stack.pushThou,

    'z':stack.squareNumbers,
    't':stack.triangleNumbers,
    'n':stack.cubeNumbers,
    'h':stack.xPowNumbers,

    'k':stack.removeOdd,
    'K':stack.removeEven,

    # Stack Commands

    '@':stack.len,
    'P':stack.pop,
    'S':stack.set,
    'm':stack.min,
    'M':stack.max,
    'c':stack.copy,
    'é':stack.half,
    'B':stack.count,
    'd':stack.pushTop,
    '`':stack.reverse,
    'Ø':stack.dupeLen,
    'e':stack.asciiset,
    'ß':stack.isSorted,
    'F':stack.fibonacci,
    'J':stack.leftShift,
    'ı':stack.pushIndex,
    'U':stack.palindrome,
    'H':stack.helloWorld,
    'j':stack.rightShift,
    'u':stack.boolConvert,
    'å':stack.concatenate,
    
    '#':stack.stack.clear,
    's':stack.stack.sort,

    # Others
    
    'Q':stack.quit,
    'q':quit_,

    'g':stack.saveVal,
    'G':stack.getVal,

    # Input and Output
    
    'o':stack.numone,
    'O':stack.numall,
    'w':stack.ascone,
    'W':stack.ascall,
    
    'ł':stack.toggleDefaultOut,

    ',':stack.setDelim,

    'I':stack.input,
    'i':stack.inputList,

    '√':stack.moveInput,
    'Ï':stack.inputRoman,
    '®':None,
    
}


class Script:

    def __init__(self,code,inputs=None):

        self.raised = 0
        self.error = None
     
        print()

        if code.count('ł') % 2 == 0:
            print('===== OUTPUT =====\n')

        tstr = False

        for i,char in enumerate(code):
            if char == '"':
                tstr = not tstr
            if char == '\n':
                if not tstr:
                    self.error = ValyrioError(msg='No newlines allowed',char=i)
        
        if code.count('//') > 1:
            self.error = ValyrioError(msg='Programs can have a maximum of one //',char=code.index('//',code.count('//')+1))

        self.code = code
        
        if hasattr(inputs,'__iter__'):
            self.input = list(inputs)[::-1]
        else:
            self.input = [inputs]

        self.raised = int(self.error != None)

        self.execute(self.code,*self.input)

        self.copy = repr(stack)
                        
        if 'N' not in code:
            stack.stack.clear()

        if not stack.suppress:
            print('\n\n==================')

    def __repr__(self):
        
        if stack.suppress:
            return ''
        
        return '''
-----Program Execution Information----
Code        : {}
Inputs      : {}
Stack       : {}
G-Variable  : {}
Byte Length : {}
Exit Status : {}
Error       : {}
--------------------------------------
'''.format(self.code,self.input,self.copy,stack.value,
           len(self.code),self.raised,self.error)

    def tokenize(self,code=None,*inputs):

        if code is None:
            code = self.code
            inputs = self.input

        temp = comp = ''
        final = []
        nums = []
        parsed = []
        instring = incomp = False

        x = 0

        for char in code:

            if char == '›':
                break
            
            if char == '-':
                try:
                    if code[x+1].isdigit():
                        temp += char
                        continue
                except:
                    pass

            if char.isdigit():
                temp += char
            else:			
                if temp:
                    nums.append(temp)
                    temp = ''
                nums.append(char)

        if temp:
            nums.append(temp)

        temp = ''

        for c in nums:
            if c == '«':
                incomp = True
            if c == '»':
                incomp = False
            if incomp:
                temp += c
            else:
                if temp:
                    final.append(temp+'»')
                    temp = ''
                else:
                    final.append(c)

        nums = final[:]
        del final[:]
        temp = ''

        for c in nums:
            if c == "'":
                instring = not instring
            if instring:
                temp += c
            else:
                if temp:
                    final.append(temp)
                    temp = ''
                else:
                    final.append(c)

        for i in final:
            if i[0] == "'":
                i += "'"
            if i and i != ' ':
                if i.isdigit() or i[0] == '-':
                    parsed.append(int(i))
                else:
                    parsed.append(i)

        return parsed, {'For':'','While':'','Inf':''}

    def execute(self,code,*inputs):

        parsed,LoopCodes = self.tokenize(code,*inputs)
        For = While = Inf = inputindex = False
        inif = True
        cont = False

        input_chars = 'Ii√Ï'

        for char in parsed:

            if cont:
                cont = False
                continue

            if char == 'y':
                cont = True

            if str(char) not in input_chars:

                if For:
                    LoopCodes['For'] += str(char)
                if While:
                    LoopCodes['While'] += str(char)
                if Inf:
                    LoopCodes['Inf'] += str(char)
                if not inif:
                    continue

            if char == 'E':
                break

            if isinstance(char,int):
                stack.push(char)
                continue

            if len(char) > 1:
                stack.push(char[1:-1])
                continue

            if char in COMMANDS and not any([For,While,Inf]):

                if char in input_chars:

                    COMMANDS[char](inputs[inputindex])
                    inputindex += 1

                elif char == '®':

                    self.execute(inputs[inputindex])
                    inputindex += 1
                    
                else:

                    COMMANDS[char]()

            elif char in SPECIALS:

                if char == '[':
                    For = True
                    LoopCodes['For'] += str(stack.pop()) + '¦'
                if char == ']':
                    For = False
                    LoopCodes['For'] = LoopCodes['For'][:-1]
                    
                if char == '{':
                    While = True
                    LoopCodes['While'] += str(int(bool(stack.pop()))) + '¦'
                if char == '}':
                    While = False
                    LoopCodes['While'] = LoopCodes['While'][:-1]
                    
                if char == '(':
                    Inf = True
                if char == ')':
                    Inf = False
                    LoopCodes['Inf'] = LoopCodes['Inf'][:-1]
                    
                if char == '¿':
                    inif = bool(stack.pop())
                if char == '?':
                    inif = True

                if char == '\\':
                    self.executeStack(stack)

            else:
                continue


            for key in LoopCodes:
                c = LoopCodes[key]
                if c:
                    
                    if key == 'For':
                        iters = int(c.split('¦')[0])
                        for i in range(iters):
                            self.execute(c.split('¦')[1])
                            
                    if key == 'Inf':
                        while True:
                            self.execute(c)
                            
                    if key == 'While':
                        confirm = stack.stack[0]
                        while confirm:
                            self.execute(c)
                            confirm = stack.stack[0]

    def executeStack(self,ExcStack):
        stackcopy = ExcStack.stack.copy()
        ExcStack.stack.clear()
        stackcopy = ''.join(reversed(list(map(chr,stackcopy))))
        self.execute(stackcopy)

prog = sys.argv[1]
inputs = map(eval, sys.argv[2:])
if prog.endswith('.txt'):
    prog = open(prog).read()

print(Script(prog,list(inputs)))
