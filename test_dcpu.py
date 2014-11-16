import dcpu
from bitarray import bitarray
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

# "additional" is a function, to which the CPU will be passed as an argument and which must evaluate to true on success. It is optional.
@pytest.mark.parametrize(("set_code_value_pairs", "get_code", "expected", "additional"), [
   ([(0x00, 0xffff)], 0x00, 0xffff, lambda cpu: cpu.reg['a'] == 0xffff),
    ([(0x01, 0xffff)], 0x01, 0xffff, lambda cpu: cpu.reg['b'] == 0xffff),
    ([(0x02, 0xffff)], 0x02, 0xffff, lambda cpu: cpu.reg['c'] == 0xffff),
    ([(0x03, 0xffff)], 0x03, 0xffff, lambda cpu: cpu.reg['x'] == 0xffff),
    ([(0x04, 0xffff)], 0x04, 0xffff, lambda cpu: cpu.reg['y'] == 0xffff),
    ([(0x05, 0xffff)], 0x05, 0xffff, lambda cpu: cpu.reg['z'] == 0xffff),
    ([(0x06, 0xffff)], 0x06, 0xffff, lambda cpu: cpu.reg['i'] == 0xffff),
    ([(0x07, 0xffff)], 0x07, 0xffff, lambda cpu: cpu.reg['j'] == 0xffff),
    ([(0x00, 0x0002), (0x08, 0xffff)], 0x08, 0xffff, lambda cpu: cpu.ram.get(0x0002) == 0xffff),
    ([(0x01, 0x0002), (0x09, 0xffff)], 0x09, 0xffff, lambda cpu: cpu.ram.get(0x0002) == 0xffff),
    ([(0x02, 0x0002), (0x0a, 0xffff)], 0x0a, 0xffff, lambda cpu: cpu.ram.get(0x0002) == 0xffff),
    ([(0x03, 0x0002), (0x0b, 0xffff)], 0x0b, 0xffff, lambda cpu: cpu.ram.get(0x0002) == 0xffff),
    ([(0x04, 0x0002), (0x0c, 0xffff)], 0x0c, 0xffff, lambda cpu: cpu.ram.get(0x0002) == 0xffff),
    ([(0x05, 0x0002), (0x0d, 0xffff)], 0x0d, 0xffff, lambda cpu: cpu.ram.get(0x0002) == 0xffff),
    ([(0x06, 0x0002), (0x0e, 0xffff)], 0x0e, 0xffff, lambda cpu: cpu.ram.get(0x0002) == 0xffff),
    ([(0x07, 0x0002), (0x0f, 0xffff)], 0x0f, 0xffff, lambda cpu: cpu.ram.get(0x0002) == 0xffff),
    #([(0x1c, 0x0010), (0x00, 0x0010), (0x08], 0x0f, 0xffff, lambda cpu: cpu.ram.get(0x0002) == 0xffff),
   ])
def test_cpu_opcodes(cpu, set_code_value_pairs, get_code, expected, additional):
    for pair in set_code_value_pairs:
        cpu.set_by_code(*pair)
    assert cpu.get_by_code(get_code) == expected
    if additional:
        assert additional(cpu)

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
def test_cpu_register_opcodes(cpu, set_code, get_code, reg):
    value = 0x0101
    cpu.set_by_code(set_code, value)
    assert cpu.get_by_code(get_code) == value
    assert cpu.reg[reg] == value

# @pytest.mark.parametrize(('set_code', 'get_code', 'reg'), [
#     (0x08, 0x08, 'a'),
#     (0x09, 0x09, 'b'),
#     (0x0a, 0x0a, 'c'),
#     (0x0b, 0x0b, 'x'),
#     (0x0c, 0x0c, 'y'),
#     (0x0d, 0x0d, 'z'),
#     (0x0e, 0x0e, 'i'),
#     (0x0f, 0x0f, 'j'),
#     ])
# def test_cpu_register_memory_opcodes(cpu, set_code, get_code, reg):
#     memloc = 0x0010
#     value = 0xfafa
#     cpu.mem.set(memloc, value)
#     cpu.reg[reg] = memloc
    
