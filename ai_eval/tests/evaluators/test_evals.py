from ai_eval.targets import NumericRangeTarget, ContainsTarget


def test_set_eval():
    
    set_eval = ContainsTarget({'a', 'b', 'c'})
    assert set_eval('a')
    assert not set_eval('d')
    
    set_eval = ContainsTarget(['a', 'b', 'c'])
    assert set_eval('a')
    assert not set_eval('d')


def test_num_range_eval():
    num_eval = NumericRangeTarget('[0, 1]')
    assert num_eval(0.5)
    assert not num_eval(1.1)
    assert num_eval(1)
    assert num_eval(0)
    
    num_eval = NumericRangeTarget((0, 1))
    assert num_eval(0.5)
    
    num_eval = NumericRangeTarget([0, 1])
    assert num_eval(0.5)
    
    num_eval = NumericRangeTarget(0.5)
    assert num_eval(0.5)
    
    num_eval = NumericRangeTarget('[0, 1)')
    assert num_eval(0.5)
    assert not num_eval(1)
    
    num_eval = NumericRangeTarget('(0, 1]')
    assert num_eval(0.5)
    assert not num_eval(0)
    
    num_eval = NumericRangeTarget('(0,)')
    assert num_eval(1)
    assert not num_eval(0)
    
    num_eval = NumericRangeTarget('(,1]')
    assert num_eval(0)
    assert num_eval(1)
    assert not num_eval(2)