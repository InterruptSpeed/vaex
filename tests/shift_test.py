import vaex
import numpy as np
import pyarrow as pa

def test_shift_basics():
    x = np.arange(4)
    df = vaex.from_arrays(x=x, y=x**2)
    dfp1 = df.shift(1, ['x'])
    dfn1 = df.shift(-1, ['x'])
    assert dfp1.x.tolist() == [None, 0, 1, 2]
    assert dfp1.y.tolist() == [0, 1, 4, 9]
    assert dfn1.x.tolist() == [1, 2, 3, None]
    assert dfn1.y.tolist() == [0, 1, 4, 9]

    assert dfp1.shift(1).x.tolist() == [None, None, 0, 1]
    assert dfp1.shift(-1).x.tolist() == [0, 1, 2, None]
    assert dfp1.shift(-1, fill_value=99).x.tolist() == [0, 1, 2, 99]

    assert dfn1.shift(1).x.tolist() == [None, 1, 2, 3]
    assert dfn1.shift(-1).x.tolist() == [2, 3, None, None]
    assert dfn1.shift(-1, fill_value=99).x.tolist() == [2, 3, None, 99]

    assert df.shift(3).x.tolist() == [None, None, None, 0]
    assert df.shift(4).x.tolist() == [None, None, None, None]
    assert df.shift(5).x.tolist() == [None, None, None, None]

    assert df.shift(-3).x.tolist() == [3, None, None, None]
    assert df.shift(-4).x.tolist() == [None, None, None, None]
    assert df.shift(-5).x.tolist() == [None, None, None, None]


def test_shift_filtered():
    x = np.array([0, 99, 1, 2, 99, 3])
    df = vaex.from_arrays(x=x, y=x**2)
    df = df[df.x != 99]
    dfp1 = df.shift(1, ['x'])
    dfn1 = df.shift(-1, ['x'])
    assert dfp1.x.tolist() == [None, 0, 1, 2]
    assert dfp1.y.tolist() == [0, 1, 4, 9]
    assert dfn1.x.tolist() == [1, 2, 3, None]
    assert dfn1.y.tolist() == [0, 1, 4, 9]

    assert dfp1.shift(1).x.tolist() == [None, None, 0, 1]
    assert dfp1.shift(-1).x.tolist() == [0, 1, 2, None]
    assert dfp1.shift(-1, fill_value=99).x.tolist() == [0, 1, 2, 99]

    assert dfn1.shift(1).x.tolist() == [None, 1, 2, 3]
    assert dfn1.shift(-1).x.tolist() == [2, 3, None, None]
    assert dfn1.shift(-1, fill_value=99).x.tolist() == [2, 3, None, 99]

    assert df.shift(3).x.tolist() == [None, None, None, 0]
    assert df.shift(4).x.tolist() == [None, None, None, None]
    assert df.shift(5).x.tolist() == [None, None, None, None]

    assert df.shift(-3).x.tolist() == [3, None, None, None]
    assert df.shift(-4).x.tolist() == [None, None, None, None]
    assert df.shift(-5).x.tolist() == [None, None, None, None]


def test_shift_string():
    x = np.arange(4)
    s = pa.array(['aap', None, 'noot', 'mies'])
    df = vaex.from_arrays(x=x, s=s)
    assert df.shift(1).s.tolist() == [None, 'aap', None, 'noot']
    assert df.shift(-1).s.tolist() == [None, 'noot', 'mies', None]
    assert df.shift(1, ['s'], fill_value='VAEX').s.tolist() == ['VAEX', 'aap', None, 'noot']
    assert df.shift(-1, ['s'], fill_value='VAEX').s.tolist() == [None, 'noot', 'mies', 'VAEX']
