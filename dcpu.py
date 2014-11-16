from enum import Enum
from functools import partial

class Opcode(Enum):
    NONBASIC = 0x0
    SET = 0x1
    ADD = 0x2
    SUB = 0x3
    MUL = 0x4
    DIV = 0x5
    MOD = 0x6
    SHL = 0x7
    SHR = 0x8
    AND = 0x9
    BOR = 0xa
    XOR = 0xb
    IFE = 0xc
    IFN = 0xd
    IFG = 0xe
    IFB = 0xf

class NonBasicOpcode(Enum):
    JSR = 0x01

def sanitized_value(value, word_length):
    if not isinstance(value, int):
        value = int(value)
    value = value % 2**word_length
    return value

def does_overflow(value, word_length):
    return not 0 <= value < 2**word_length

# helper method to construct valid 16-bit op word
def compile_word(b, a, o):
    return o + (a << 4) + (b << 10)

# returns (b, a, o)
def decompile_word(word):
    return word >> 10, (word >> 4) & 0b000000111111, word & 0b0000000000001111

class RAM():
    # initial_contents must be a list of words!
    def __init__(self, word_length, size, initial_contents=None):
        self.word_length = word_length
        self.contents = [0x0000 for _ in range(size)]
        if initial_contents:
            assert len(initial_contents) <= len(self.contents)
            self.contents[:len(initial_contents)] = initial_contents

    @property
    def size(self):
        return len(self.contents)

    def get(self, pos):
        return self.contents[pos]

    def set(self, pos, value):
        value = sanitized_value(value, self.word_length)
        self.contents[pos] = value

class DCPURegisterBank():

    all_regs = ('a', 'b', 'c', 'x', 'y', 'z', 'i', 'j', 'pc', 'sp', 'o')
    regs = all_regs[:8]

    def __init__(self, word_length, values):
        self.word_length = word_length
        for reg in self.all_regs:
            self[reg] = values[reg]

    def __len__(self):
        return len(self.all_regs)

    def __getitem__(self, key):
        if key in self.all_regs:
            return getattr(self, key)
        else:
            raise KeyError

    def __setitem__(self, key, value):
        if key in self.all_regs:
            value = sanitized_value(value, self.word_length)
            setattr(self, key, value)
        else:
            raise KeyError

    def __iter__(self):
        return iter(self.all_regs)

    def __contains__(self, key):
        return key in self.all_regs

    def items(self):
        for key in self.all_regs:
            return (key, getattr(self, key))

class CPU():
    # initial_registers must be a dictionary with a, b, c, x, y, z, i, j, pc, sp, o.
    def __init__(self, initial_registers=None, initial_ram=None, initial_cycle=0):
        if not initial_ram:
            initial_ram = RAM(word_length=16, size=2**16)

        if not initial_registers:
            initial_registers = {key: 0x0000 for key in DCPURegisterBank.all_regs}

        self.reg = DCPURegisterBank(word_length=16, values=initial_registers)
        self.ram = initial_ram
        self.cycle = initial_cycle

        # TODO: maybe change these to simply pass the opcode into the lambda, instead of using partials?

        self.value_codes = {}
        self.value_codes.update({x: partial(lambda y: self.reg[self.reg.regs[y]], x) for x in range(0x00, 0x08)})
        self.value_codes.update({x + 0x08: partial(lambda y: self.ram.get(self.reg[self.reg.regs[y]]), x) for x in range(0x00, 0x08)})
        self.value_codes.update({x + 0x10: partial(lambda y: self.ram.get(self.next_word() + self.reg[self.reg.regs[y]]), x) for x in range(0x00, 0x08)})

        self.value_codes.update({
            0x18: lambda: self.pop(),
            0x19: lambda: self.ram.get(self.reg.sp),
            0x1a: lambda: self.push(),
            0x1b: lambda: self.reg.sp,
            0x1c: lambda: self.reg.pc,
            0x1d: lambda: self.reg.o,
            0x1e: lambda: self.ram.get(self.next_word()),
            0x1f: lambda: self.next_word(),
        })

        for x in range(0x20):
            self.value_codes[x + 0x20] = partial(lambda y: y, x)

        self.set_codes = {}
        self.set_codes.update({x: partial(lambda y, value: self.reg.__setitem__(self.reg.regs[y], value), x) for x in range(0x00, 0x08)})
        self.set_codes.update({x + 0x08: partial(lambda y, value: self.ram.set(self.reg[self.reg.regs[y]], value), x) for x in range(0x00, 0x08)})
        self.set_codes.update({x + 0x10: partial(lambda y, value: self.ram.set(self.next_word() + self.reg[self.reg.regs[y]], value), x) for x in range(0x00, 0x08)})

        self.set_codes.update({
            0x18: lambda value: self.pop(value),
            0x19: lambda value: self.ram.set(self.reg.sp, value),
            0x1a: lambda value: self.push(value),
            0x1b: lambda value: self.reg.__setitem__('sp', value),
            0x1c: lambda value: self.reg.__setitem__('pc', value),
            0x1d: lambda value: self.reg.__setitem__('o', value),
            0x1e: lambda value: self.ram.set(self.next_word(), value),
            0x1f: lambda value: None,
        })

        for x in range(0x20):
            self.set_codes[x + 0x20] = lambda value: None

        self.opcodes = {
            Opcode.SET: self.SET,
            Opcode.ADD: self.ADD,
            Opcode.SUB: self.SUB,
            Opcode.MUL: self.MUL,
            Opcode.DIV: self.DIV,
            Opcode.MOD: self.MOD,
            }

    def next_word(self):
        word = self.ram.get(self.reg.pc)
        self.reg.pc += 1
        return word

    def pop(self, value=None):
        if value:
            self.ram.set(self.reg.sp, value)
        else:
            value = self.ram.get(self.reg.sp)
        self.reg.sp += 1
        return value

    def push(self, value=None):
        self.reg.sp -= 1
        if value:
            self.ram.set(self.reg.sp, value)
        else:
            value = self.ram.get(self.reg.sp)
        return value

    def get_by_code(self, code):
        if (0x10 <= code <= 0x17) or code in [0x1e, 0x1f]:
            self.cycle += 1
        return self.value_codes[code]()

    def set_by_code(self, code, value):
        if (0x10 <= code <= 0x17) or code in [0x1e, 0x1f]:
            self.cycle += 1
        self.set_codes[code](value)

    def step(self):
        # read PC
        word = self.next_word()
        b, a, o = decompile_word(word)
        opcode = Opcode(o)
        if opcode == Opcode.NONBASIC:
            opcode = NonBasicOpcode(a)

        # execute instruction
        self.execute_op(opcode, a, b)

    def execute_op(self, opcode, a, b):
        self.opcodes[opcode](a, b)

    def SET(self, a, b):
        self.cycle += 1
        self.set_by_code(a, self.get_by_code(b))

    def ADD(self, a, b):
        self.cycle += 2
        value = self.get_by_code(a) + self.get_by_code(b)
        self.set_by_code(a, value)
        # technically, this doesn't support the case that ram and regs are different word lengths.
        self.reg.o = 0 if value < 2**self.reg.word_length else 0x0001

    def SUB(self, a, b):
        self.cycle += 2
        value = self.get_by_code(a) - self.get_by_code(b)
        self.set_by_code(a, value)
        # technically, this doesn't support the case that ram and regs are different word lengths.
        self.reg.o = 0 if value >= 0 else 0xffff

    def MUL(self, a, b):
        self.cycle += 2
        a_val, b_val = self.get_by_code(a), self.get_by_code(b)
        self.set_by_code(a, a_val * b_val)
        self.reg.o = ((a_val*b_val)>>16)&0xffff

    def DIV(self, a, b):
        self.cycle += 3
        a_val, b_val = self.get_by_code(a), self.get_by_code(b)
        try:
            self.set_by_code(a, a_val // b_val)
            self.reg.o = ((a_val<<16)//b_val)&0xffff
        except ZeroDivisionError:
            self.set_by_code(a, 0)

    def MOD(self, a, b):
        self.cycle += 3
        a_val, b_val = self.get_by_code(a), self.get_by_code(b)
        try:
            self.set_by_code(a, a_val % b_val)
        except ZeroDivisionError:
            self.set_by_code(a, 0)
