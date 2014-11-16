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

    def __getitem__(self, key):
        if key in self.all_regs:
            return getattr(self, key)
        else:
            raise KeyError

    def __setitem__(self, key, value):
        if key in self.all_regs:
            setattr(self, key, value)
        else:
            raise KeyError

    def __iter__(self):
        return iter(self.all_regs)

    def __setattr__(self, name, value):
        if name in self.all_regs:
            value = sanitized_value(value, self.word_length)
        super().__setattr__(name, value)

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

        self.operands = {}
        self.operands.update({x: lambda code: self.reg.regs[code] for x in range(0x00, 0x08)})
        self.operands.update({x + 0x08: lambda code: self.reg[self.reg.regs[code - 0x08]] for x in range(0x00, 0x08)})
        self.operands.update({x + 0x10: lambda code: self.next_word() + self.reg[self.reg.regs[code - 0x10]] for x in range(0x00, 0x08)})

        self.operands.update({
            0x18: lambda code: self.pop_addr(),
            0x19: lambda code: self.peek_addr(),
            0x1a: lambda code: self.push_addr(),
            0x1b: lambda code: 'sp',
            0x1c: lambda code: 'pc',
            0x1d: lambda code: 'o',
            0x1e: lambda code: self.next_word(),
        })

    def next_word(self):
        word = self.ram.get(self.reg.pc)
        self.reg.pc += 1
        return word

    def pop_addr(self):
        address = self.reg.sp
        self.reg.sp += 1
        return address

    def push_addr(self):
        self.reg.sp -= 1
        return self.reg.sp

    def peek_addr(self):
        return self.reg.sp

    def needs_next_word(self, operand):
        return (0x10 <= operand <= 0x17) or operand in [0x1e, 0x1f]

    # This has side effects (it can increment PC or affect SP)
    def address_for_operand(self, operand):
        if self.needs_next_word(operand):
            self.cycle += 1
        try:
            return self.operands[operand](operand)
        except KeyError:
            return None

    # This has no side effects except in the case of code=0x1f, in which case PC++
    def get_by_address(self, address, code=None):
        if address is None and code is not None:
            if code == 0x1f:
                return self.next_word()
            elif 0x20 <= code <= 0x3f:
                return code - 0x20
        elif isinstance(address, int):
            return self.ram.get(address)
        else:
            return self.reg[address]

    def set_by_address(self, address, value):
        if isinstance(address, int):
            self.ram.set(address, value)
        else:
            self.reg[address] = value

    def get_by_code(self, code, return_addr=False):
        addr = self.address_for_operand(code)
        value = self.get_by_address(addr, code=code)
        return value if not return_addr else (value, addr)

    def set_by_code(self, code, value):
        addr = self.address_for_operand(code)
        if addr:
            self.set_by_address(addr, value)
        else:
            pass

    def step(self):
        word = self.next_word()
        b, a, o = decompile_word(word)

        opcode = Opcode(o)
        if opcode == Opcode.NONBASIC:
            opcode = NonBasicOpcode(a)
            a, b = b, None

        a_val, addr = self.get_by_code(a, return_addr=True)
        operation = getattr(self, opcode.name)
        if not isinstance(opcode, NonBasicOpcode):
            b_val = self.get_by_code(b)
            operation(a_val, b_val, addr)
        else:
            operation(a_val, addr)

    def SET(self, a, b, addr):
        self.cycle += 1
        self.set_by_address(addr, b)

    def ADD(self, a, b, addr):
        self.cycle += 2
        value = a + b
        self.set_by_address(addr, value)
        self.reg.o = 0 if value < 2**16 else 0x0001

    def SUB(self, a, b, addr):
        self.cycle += 2
        value = a - b
        self.set_by_address(addr, value)
        self.reg.o = 0 if value >= 0 else 0xffff

    def MUL(self, a, b, addr):
        self.cycle += 2
        self.set_by_address(addr, a*b)
        self.reg.o = ((a*b)>>16)&0xffff

    def DIV(self, a, b, addr):
        self.cycle += 3
        try:
            self.set_by_address(addr, a // b)
            self.reg.o = ((a<<16)//b)&0xffff
        except ZeroDivisionError:
            self.set_by_address(addr, 0)

    def MOD(self, a, b, addr):
        self.cycle += 3
        try:
            self.set_by_address(addr, a % b)
        except ZeroDivisionError:
            self.set_by_address(addr, 0)

    def SHL(self, a, b, addr):
        self.cycle += 2
        self.set_by_address(addr, a<<b)
        self.reg.o = ((a<<b)>>16)&0xffff

    def SHR(self, a, b, addr):
        self.cycle += 2
        self.set_by_address(addr, a>>b)
        self.reg.o = ((a<<16)>>b)&0xffff

    def AND(self, a, b, addr):
        self.cycle += 1
        self.set_by_address(addr, a & b)

    def BOR(self, a, b, addr):
        self.cycle += 1
        self.set_by_address(addr, a | b)

    def XOR(self, a, b, addr):
        self.cycle += 1
        self.set_by_address(addr, a ^ b)

    def IFE(self, a, b, addr):
        self.cycle += 2
        if a == b:
            pass
        else:
            self.skip_next_and_cycle()

    def IFN(self, a, b, addr):
        self.cycle += 2
        if a != b:
            pass
        else:
            self.skip_next_and_cycle()

    def IFG(self, a, b, addr):
        self.cycle += 2
        if a > b:
            pass
        else:
            self.skip_next_and_cycle()

    def IFB(self, a, b, addr):
        self.cycle += 2
        if a & b != 0:
            pass
        else:
            self.skip_next_and_cycle()

    def skip_next_and_cycle(self):
        word = self.next_word() # this increments PC!
        b, a, o = decompile_word(word)
        if self.needs_next_word(a):
            self.next_word()
        if self.needs_next_word(b):
            self.next_word()
        self.cycle += 1

    def JSR(self, a, addr):
        self.cycle += 2
        addr = self.push_addr()
        self.set_by_address(addr, self.reg.pc)
        self.reg.pc = a
