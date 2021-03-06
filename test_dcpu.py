import dcpu
from dcpu import compile_word, decompile_word
import pytest

@pytest.fixture
def cpu():
    return dcpu.CPU()

@pytest.fixture
def ram():
    return dcpu.RAM(word_length=16, size=0x10000)

def test_ram_init():
    assert len(dcpu.RAM(word_length=8, size=0x1000).contents) == 0x1000
    assert len(dcpu.RAM(word_length=16, size=0x20000).contents) == 0x20000
    ram = dcpu.RAM(word_length=16, size=0x1000, initial_contents=[0, 1, 2, 3])
    assert ram.get(0) == 0
    assert ram.get(1) == 1
    assert ram.get(2) == 2
    assert ram.get(3) == 3
    assert ram.get(4) == 0

def test_ram_set(ram):
    ram.set(0x01, 0xffff) # maximum word length
    assert ram.get(0x01) == 0xffff
    with pytest.raises(TypeError):
        ram.set(0x01, pytest)
    ram.set(0x01, 2**ram.word_length + 0x10) # more than maximum word length
    assert ram.get(0x01) == (2**ram.word_length + 0x10) % 2**ram.word_length
    ram.set(0x01, -0x10)
    assert ram.get(0x01) == -0x10 % 2**ram.word_length
    with pytest.raises(IndexError):
        ram.set(0xffffffff, 0x0000)

def test_cpu_init():
    cpu = dcpu.CPU()
    for reg in cpu.reg:
        assert cpu.reg[reg] == 0
    assert cpu.ram.word_length == 16
    assert cpu.ram.size == 0x10000
    assert cpu.ram.get(0) == 0

    initial_ram = dcpu.RAM(word_length=16, size=0x10000, initial_contents=[0x0000, 0x0000, 0xffff, 0xffff, 0x0001])
    initial_registers = {'a': 0x1000,
                         'b': 0x2000,
                         'c': 0x3000,
                         'x': 0x4000,
                         'y': 0x5000,
                         'z': 0x6000,
                         'i': 0x7000,
                         'j': 0x8000,
                         'pc': 0x9000,
                         'sp': 0xa000,
                         'o': 0x1,}
    initial_cycle = 2
    cpu = dcpu.CPU(initial_registers=initial_registers, initial_ram=initial_ram, initial_cycle=initial_cycle)

    for reg in cpu.reg:
        assert cpu.reg[reg] == initial_registers[reg]
    assert cpu.cycle == initial_cycle

    assert cpu.ram.get(0) == 0x0000
    assert cpu.ram.get(1) == 0x0000
    assert cpu.ram.get(2) == 0xffff
    assert cpu.ram.get(3) == 0xffff
    assert cpu.ram.get(4) == 0x0001
    assert cpu.ram.get(5) == 0x0000

def test_register_bank():
    initial_registers = {'a': 0x1000,
                         'b': 0x2000,
                         'c': 0x3000,
                         'x': 0x4000,
                         'y': 0x5000,
                         'z': 0x6000,
                         'i': 0x7000,
                         'j': 0x8000,
                         'pc': 0x9000,
                         'sp': 0xa000,
                         'o': 0x1,}
    word_length = 16
    regbank = dcpu.DCPURegisterBank(word_length=word_length, values=initial_registers)

    test_value = 0x1010

    for reg in initial_registers:
        assert regbank[reg] == initial_registers[reg]
        assert getattr(regbank, reg) == initial_registers[reg]
        assert regbank.__getitem__(reg) == initial_registers[reg]
        regbank[reg] = test_value
        assert regbank[reg] == test_value
        regbank[reg] = 2**word_length + 0x10 # more than maximum word length
        assert regbank[reg] == (2**word_length + 0x10) % 2**word_length
        regbank[reg] = -1
        assert regbank[reg] == -1 % 2**word_length

    with pytest.raises(KeyError):
        regbank['nonsense'] = test_value

    with pytest.raises(KeyError):
        assert regbank['nonsense'] == 0x0000

    regbank.pc = 0xffff
    regbank.pc += 1
    assert regbank.pc == 0x0000

@pytest.mark.parametrize(('set_code', 'get_code', 'reg'), [
    (0x00, 0x00, 'a'),
    (0x01, 0x01, 'b'),
    (0x02, 0x02, 'c'),
    (0x03, 0x03, 'x'),
    (0x04, 0x04, 'y'),
    (0x05, 0x05, 'z'),
    (0x06, 0x06, 'i'),
    (0x07, 0x07, 'j'),
    ])
def test_cpu_register_operand_codes(cpu, set_code, get_code, reg):
    value = 0x0101
    cpu.set_by_code(set_code, value)
    assert cpu.get_by_code(get_code) == value
    assert cpu.reg[reg] == value

@pytest.mark.parametrize(('set_code', 'get_code', 'reg'), [
    (0x08, 0x08, 'a'),
    (0x09, 0x09, 'b'),
    (0x0a, 0x0a, 'c'),
    (0x0b, 0x0b, 'x'),
    (0x0c, 0x0c, 'y'),
    (0x0d, 0x0d, 'z'),
    (0x0e, 0x0e, 'i'),
    (0x0f, 0x0f, 'j'),
    ])
def test_cpu_register_memory_operand_codes(cpu, set_code, get_code, reg):
    memloc = 0x0010
    value = 0xfafa
    cpu.reg[reg] = memloc
    cpu.set_by_code(set_code, value)
    assert cpu.ram.get(memloc) == value
    assert cpu.get_by_code(get_code) == value

@pytest.mark.parametrize(('set_code', 'get_code', 'reg'), [
    (0x10, 0x10, 'a'),
    (0x11, 0x11, 'b'),
    (0x12, 0x12, 'c'),
    (0x13, 0x13, 'x'),
    (0x14, 0x14, 'y'),
    (0x15, 0x15, 'z'),
    (0x16, 0x16, 'i'),
    (0x17, 0x17, 'j'),
    ])
def test_cpu_register_memory_operand_codes(cpu, set_code, get_code, reg):
    memloc = 0x0020
    value = 0xfafa
    # set PC to an arbitrary value
    pc_start = 0x0002
    cpu.reg['pc'] = pc_start
    # set next word to arbitrary small value
    next_word = 0x0010
    cpu.ram.set(0x0002, 0x0010)
    cpu.reg[reg] = memloc - next_word
    cpu.set_by_code(set_code, value)
    assert cpu.ram.get(memloc) == value
    assert cpu.reg['pc'] == pc_start + 1
    # because set in this case mutates state, we need to reset pc before get
    cpu.reg['pc'] = pc_start
    assert cpu.get_by_code(get_code) == value
    assert cpu.reg['pc'] == pc_start + 1

def test_cpu_pop_peek_push_operand_codes(cpu):
    assert cpu.reg['sp'] == 0x0000
    cpu.set_by_code(0x1a, 0x0010)
    cpu.set_by_code(0x1a, 0x0020)
    cpu.set_by_code(0x1a, 0x0030)
    cpu.set_by_code(0x1a, 0x0011)
    assert cpu.reg['sp'] == (0x0000 - 4) % 2**cpu.reg.word_length
    assert cpu.get_by_code(0x19) == 0x0011
    cpu.set_by_code(0x19, 0x0040)
    assert cpu.get_by_code(0x19) == 0x0040
    assert cpu.reg['sp'] == (0x0000 - 4) % 2**cpu.reg.word_length
    assert cpu.get_by_code(0x18) == 0x0040
    assert cpu.reg['sp'] == (0x0000 - 3) % 2**cpu.reg.word_length
    assert cpu.get_by_code(0x18) == 0x0030
    assert cpu.reg['sp'] == (0x0000 - 2) % 2**cpu.reg.word_length
    cpu.set_by_code(0x18, 0x0011)
    assert cpu.reg['sp'] == (0x0000 - 1) % 2**cpu.reg.word_length
    assert cpu.ram.get(cpu.reg.sp - 1) == 0x0011
    assert cpu.get_by_code(0x1a) == 0x0011
    assert cpu.reg['sp'] == (0x0000 - 2) % 2**cpu.reg.word_length

def test_sp_pc_o_operand_codes(cpu):
    cpu.set_by_code(0x1b, 0x0001)
    assert cpu.reg.sp == 0x0001
    assert cpu.get_by_code(0x1b) == 0x0001
    cpu.set_by_code(0x1c, 0x0002)
    assert cpu.reg.pc == 0x0002
    assert cpu.get_by_code(0x1c) == 0x0002
    cpu.set_by_code(0x1d, 0x0003)
    assert cpu.reg.o == 0x0003
    assert cpu.get_by_code(0x1d) == 0x0003

def test_next_word_operand_codes(cpu):
    assert cpu.reg.pc == 0x0000
    cpu.ram.set(0x0000, 0x0010)
    cpu.ram.set(0x0001, 0x0020)
    cpu.set_by_code(0x1e, 0x0030)
    assert cpu.ram.get(0x0010) == 0x0030
    assert cpu.reg.pc == 0x0001
    # should do nothing
    cpu.set_by_code(0x1f, 0x0040)
    assert cpu.reg.pc == 0x0001
    assert cpu.ram.get(0x0010) == 0x0030
    assert cpu.ram.get(0x0000) == 0x0010
    assert cpu.ram.get(0x0001) == 0x0020
    assert cpu.get_by_code(0x1f) == 0x0020
    assert cpu.reg.pc == 0x0002

def test_literal_operand_codes(cpu):
    for x in range(0x00, 0x20):
        assert cpu.get_by_code(x + 0x20) == x
        cpu.set_by_code(x + 0x20, 0xffff)
        assert cpu.get_by_code(x + 0x20) == x

def test_compile_decompile_word():
    assert compile_word(0x00, 0x00, 0x0) == 0x0000
    assert compile_word(0x03, 0x01, 0x2) == 0b0000110000010010 # ADD register B to register X and put in register B
    assert decompile_word(0b0000110000010010) == (0x03, 0x01, 0x2)

def test_SET(cpu):
    assert cpu.reg.b == 0x0000
    cpu.ram.set(0x0000, compile_word(0x22, 0x01, 0x1)) # set reg b to literal 2
    cpu.step()
    assert cpu.reg.b == 0x0002
    assert cpu.cycle == 1
    assert cpu.reg.pc == 1

def test_ADD(cpu):
    cpu.reg.b = 0x0004
    cpu.ram.set(0x0000, compile_word(0x22, 0x01, 0x2)) # set reg b to literal 2 + 4
    cpu.step()
    assert cpu.reg.b == 0x0006
    assert cpu.cycle == 2
    assert cpu.reg.pc == 1
    assert cpu.reg.o == 0

def test_ADD_o(cpu):
    cpu.reg.a = 0xf000
    cpu.reg.b = 0x2000
    cpu.ram.set(0x0000, compile_word(0x01, 0x00, 0x2)) # set reg a to a + b
    cpu.step()
    assert cpu.reg.a == 0x1000
    assert cpu.reg.b == 0x2000
    assert cpu.cycle == 2
    assert cpu.reg.pc == 1
    assert cpu.reg.o == 1

def test_SUB(cpu):
    cpu.reg.b = 0x0005
    cpu.ram.set(0x0000, compile_word(0x22, 0x01, 0x3)) # set reg b to 5 - literal 2
    cpu.step()
    assert cpu.reg.b == 0x0003
    assert cpu.cycle == 2
    assert cpu.reg.pc == 1
    assert cpu.reg.o == 0

def test_SUB_o(cpu):
    cpu.reg.a = 0x1000
    cpu.reg.b = 0xf000
    cpu.ram.set(0x0000, compile_word(0x01, 0x00, 0x3)) # set reg a to a - b
    cpu.step()
    assert cpu.reg.a == 0x2000
    assert cpu.reg.b == 0xf000
    assert cpu.cycle == 2
    assert cpu.reg.pc == 1
    assert cpu.reg.o == 0xffff

def test_MUL(cpu):
    cpu.reg.b = 0x0004
    cpu.ram.set(0x0000, compile_word(0x22, 0x01, 0x4)) # set reg b to literal 2 * 4
    cpu.step()
    assert cpu.reg.b == 0x0008
    assert cpu.cycle == 2
    assert cpu.reg.pc == 1
    assert cpu.reg.o == 0

def test_MUL_o(cpu):
    cpu.reg.a = 0x02ff
    cpu.reg.b = 0x00ff
    cpu.ram.set(0x0000, compile_word(0x01, 0x00, 0x4)) # set reg a to a * b
    cpu.step()
    assert cpu.reg.a == 0xfc01
    assert cpu.cycle == 2
    assert cpu.reg.pc == 1
    assert cpu.reg.o == 0x0002

def test_DIV(cpu):
    cpu.reg.b = 0x0009
    cpu.ram.set(0x0000, compile_word(0x23, 0x01, 0x5)) # set reg b to 9 // literal 3
    cpu.step()
    assert cpu.reg.b == 0x0003
    assert cpu.cycle == 3
    assert cpu.reg.pc == 1
    assert cpu.reg.o == 0

def test_DIV_o(cpu):
    cpu.reg.b = 0x0009
    cpu.ram.set(0x0000, compile_word(0x22, 0x01, 0x5)) # set reg b to 9 // literal 2
    cpu.step()
    assert cpu.reg.b == 0x0004
    assert cpu.cycle == 3
    assert cpu.reg.pc == 1
    assert cpu.reg.o == 0x8000

def test_DIV_zero(cpu):
    cpu.reg.b = 0x0009
    cpu.ram.set(0x0000, compile_word(0x20, 0x01, 0x5)) # set reg b to 9 // literal 0
    cpu.step()
    assert cpu.reg.b == 0x0000
    assert cpu.cycle == 3
    assert cpu.reg.pc == 1
    assert cpu.reg.o == 0x0000

def test_MOD(cpu):
    cpu.reg.b = 0x0009
    cpu.ram.set(0x0000, compile_word(0x22, 0x01, 0x6)) # set reg b to 9 % literal 2
    cpu.step()
    assert cpu.reg.b == 0x0001
    assert cpu.cycle == 3
    assert cpu.reg.pc == 1
    assert cpu.reg.o == 0x0000

def test_MOD_zero(cpu):
    cpu.reg.b = 0x0009
    cpu.ram.set(0x0000, compile_word(0x20, 0x01, 0x6)) # set reg b to 9 % literal 0
    cpu.step()
    assert cpu.reg.b == 0x0000
    assert cpu.cycle == 3
    assert cpu.reg.pc == 1
    assert cpu.reg.o == 0x0000

def test_SHL(cpu):
    cpu.reg.a = 0x0009
    cpu.ram.set(0x0000, compile_word(0x22, 0x00, 0x7)) # set reg a to 9 << 2
    cpu.step()
    assert cpu.reg.a == 0x0024
    assert cpu.cycle == 2
    assert cpu.reg.pc == 1
    assert cpu.reg.o == 0x0000

def test_SHL_o(cpu):
    cpu.reg.a = 0xffff
    cpu.ram.set(0x0000, compile_word(0x22, 0x00, 0x7)) # set reg a to 0xffff << 2
    cpu.step()
    assert cpu.reg.a == 0xfffc
    assert cpu.cycle == 2
    assert cpu.reg.pc == 1
    assert cpu.reg.o == 0x0003

def test_SHR(cpu):
    cpu.reg.a = 0xff00
    cpu.ram.set(0x0000, compile_word(0x24, 0x00, 0x8)) # set reg a to 0xff00 >> 4
    cpu.step()
    assert cpu.reg.a == 0x0ff0
    assert cpu.cycle == 2
    assert cpu.reg.pc == 1
    assert cpu.reg.o == 0x0000

def test_SHL_o(cpu):
    cpu.reg.a = 0xff00
    cpu.ram.set(0x0000, compile_word(0x2c, 0x00, 0x8)) # set reg a to 0xff00 >> 12
    cpu.step()
    assert cpu.reg.a == 0x000f
    assert cpu.cycle == 2
    assert cpu.reg.pc == 1
    assert cpu.reg.o == 0xf000

def test_AND(cpu):
    cpu.reg.a = 0x0003
    cpu.ram.set(0x0000, compile_word(0x21, 0x00, 0x9)) # set reg a to 0b11 & 0b01
    cpu.step()
    assert cpu.reg.a == 0x0001
    assert cpu.cycle == 1
    assert cpu.reg.pc == 1

def test_BOR(cpu):
    cpu.reg.a = 0x0002
    cpu.ram.set(0x0000, compile_word(0x21, 0x00, 0xa)) # set reg a to 0b10 | 0b01
    cpu.step()
    assert cpu.reg.a == 0x0003
    assert cpu.cycle == 1
    assert cpu.reg.pc == 1

def test_XOR(cpu):
    cpu.reg.a = 0x0003
    cpu.ram.set(0x0000, compile_word(0x21, 0x00, 0xb)) # set reg a to 0b11 ^ 0b01
    cpu.step()
    assert cpu.reg.a == 0x0002
    assert cpu.cycle == 1
    assert cpu.reg.pc == 1

def test_IFE(cpu):
    cpu.reg.a = 0x0001
    cpu.ram.set(0x0000, compile_word(0x21, 0x00, 0xc)) # skip next instruction unless a == 1
    cpu.step()
    assert cpu.cycle == 2
    assert cpu.reg.pc == 1

    cpu.ram.set(0x0001, compile_word(0x22, 0x00, 0xc)) # skip next instruction unless a == 2
    cpu.ram.set(0x0002, 0x7803) # sub A, [next_word]
    cpu.ram.set(0x0003, 0x1000) # aforementioned next word
    cpu.step()
    assert cpu.cycle == 5
    assert cpu.reg.pc == 4 # must skip BOTH words of next command

def test_IFN(cpu):
    cpu.reg.a = 0x0002
    cpu.ram.set(0x0000, compile_word(0x21, 0x00, 0xd)) # skip next instruction unless a != 1
    cpu.step()
    assert cpu.cycle == 2
    assert cpu.reg.pc == 1

    cpu.ram.set(0x0001, compile_word(0x22, 0x00, 0xd)) # skip next instruction unless a != 2
    cpu.ram.set(0x0002, 0x7803) # sub A, [next_word]
    cpu.ram.set(0x0003, 0x1000) # aforementioned next word
    cpu.step()
    assert cpu.cycle == 5
    assert cpu.reg.pc == 4 # must skip BOTH words of next command

def test_IFG(cpu):
    cpu.reg.a = 0x0002
    cpu.ram.set(0x0000, compile_word(0x21, 0x00, 0xe)) # skip next instruction unless a > 1
    cpu.ram.set(0x0002, 0x7803) # sub A, [next_word]
    cpu.ram.set(0x0003, 0x1000) # aforementioned next word
    cpu.step()
    assert cpu.cycle == 2
    assert cpu.reg.pc == 1

    cpu.ram.set(0x0001, compile_word(0x22, 0x00, 0xe)) # skip next instruction unless a > 2
    cpu.ram.set(0x0002, 0x7803) # sub A, [next_word]
    cpu.ram.set(0x0003, 0x1000) # aforementioned next word
    cpu.step()
    assert cpu.cycle == 5
    assert cpu.reg.pc == 4 # must skip BOTH words of next command

def test_IFB(cpu):
    cpu.reg.a = 0x0001
    cpu.ram.set(0x0000, compile_word(0x21, 0x00, 0xf)) # skip next instruction unless a & 1 != 0
    cpu.step()
    assert cpu.cycle == 2
    assert cpu.reg.pc == 1

    cpu.ram.set(0x0001, compile_word(0x22, 0x00, 0xf)) # skip next instruction unless a & 2 != 0
    cpu.ram.set(0x0002, 0x7803) # sub A, [next_word]
    cpu.ram.set(0x0003, 0x1000) # aforementioned next word
    cpu.step()
    assert cpu.cycle == 5
    assert cpu.reg.pc == 4 # must skip BOTH words of next command

def test_JSR(cpu):
    cpu.ram.set(0x0000, compile_word(0x25, 0x01, 0x00)) # push address of next instruction to stack and jump to 5
    cpu.step()
    assert cpu.cycle == 2
    assert cpu.reg.pc == 5
    assert cpu.ram.get(cpu.reg.sp) == 1

def test_a_handled_before_b(cpu):
    cpu.ram.set(0x0000, 0x7de1) # SET [next_word], next_word
    cpu.ram.set(0x0001, 0x1000)
    cpu.ram.set(0x0002, 0x0020)
    cpu.step()
    assert cpu.reg.pc == 3
    assert cpu.cycle == 3
    assert cpu.ram.get(0x1000) == 0x0020
    assert cpu.ram.get(0x0020) != 0x1000

def test_example_code():
    contents = [0x7c01, 0x0030, 0x7de1, 0x1000,
                0x0020, 0x7803, 0x1000, 0xc00d,
                0x7dc1, 0x001a, 0xa861, 0x7c01,
                0x2000, 0x2161, 0x2000, 0x8463,
                0x806d, 0x7dc1, 0x000d, 0x9031,
                0x7c10, 0x0018, 0x7dc1, 0x001a,
                0x9037, 0x61c1, 0x7dc1, 0x001a]
    ram = dcpu.RAM(word_length=16, size=0x10000, initial_contents=contents)
    cpu = dcpu.CPU(initial_ram=ram)

    cpu.step()
    assert cpu.reg.a == 0x30
    assert cpu.cycle == 2
    assert cpu.reg.pc == 2

    cpu.step()
    assert cpu.ram.get(0x1000) == 0x0020
    assert cpu.ram.get(0x0020) == 0x0000
    assert cpu.cycle == 5
    assert cpu.reg.pc == 5

    cpu.step()
    assert cpu.reg.a == 0x0010
    assert cpu.cycle == 8
    assert cpu.reg.pc == 7

    cpu.step()
    assert cpu.cycle == 11
    assert cpu.reg.pc == 0xa

    cpu.step()
    assert cpu.reg.i == 10
    assert cpu.cycle == 12
    assert cpu.reg.pc == 0xb

    cpu.step()
    assert cpu.reg.a == 0x2000
    assert cpu.cycle == 14
    assert cpu.reg.pc == 0x000d

    for _ in range(4): cpu.step()
    assert cpu.reg.i == 9
    assert cpu.cycle == 22
    assert cpu.reg.pc == 0x000d

    for _ in range(4 * 8): cpu.step()
    assert cpu.reg.i == 1
    assert cpu.cycle == 86
    assert cpu.reg.pc == 0x000d

    for _ in range(3): cpu.step()
    assert cpu.reg.i == 0
    assert cpu.cycle == 93
    assert cpu.reg.pc == 0x0013

    cpu.step()
    assert cpu.reg.x == 0x4
    assert cpu.reg.pc == 0x0014
    assert cpu.cycle == 94

    cpu.step()
    assert cpu.reg.sp == 0xffff
    assert cpu.ram.get(0xffff) == 0x0016
    assert cpu.reg.pc == 0x0018
    assert cpu.cycle == 97

    cpu.step()
    assert cpu.reg.x == 0x0040
    assert cpu.cycle == 99
    assert cpu.reg.pc == 0x0019

    cpu.step()
    assert cpu.reg.sp == 0x0000
    assert cpu.reg.pc == 0x0016
    assert cpu.cycle == 100

    cpu.step()
    assert cpu.reg.pc == 0x001a
    assert cpu.cycle == 102

    for _ in range(100): cpu.step()
    assert cpu.reg.pc == 0x001a
    assert cpu.cycle == 302
