"""
The interpreter plus standard library. Design goals:

  * Keep core.py self-contained and unit testable.
  * Extend the primitives, root object, and environment 
    with conveniences written conveniently in interpreted code.
  * Avoid slowing down the core.py primitives.

We do this by mutating the core.py method tables, which is icky.
Maybe doing everything by extension wouldn't hurt speed much, though I
haven't tried it.

TODO: add some interpreted miranda methods too.
"""

import core

def sys_load(name):
    bob = sys_bob[name]
    extend_in_place(bob, load('sys_%s.bicicleta' % name))

def extend_in_place(bob, overlay):
    # TODO: deep copy? Shallow is all we need for now.
    for slot, method in overlay.methods.items():
        assert slot not in bob
        bob.methods[slot] = method

def load(filename):
    return core.parse(open(filename).read()).eval(global_env)

sys_bob = core.Prim(None, {
    'true':  lambda _, me: core.true_claim,
    'false': lambda _, me: core.false_claim,
})
global_env = {'sys': sys_bob}

sys_load('true')
sys_load('false')
extend_in_place(core.Number(42), load('sys_number.bicicleta'))
extend_in_place(core.String('hi'), load('sys_string.bicicleta'))
extend_in_place(sys_bob, load('sys.bicicleta'))

def run(program, prim=repr):
    return core.parse(program).eval(global_env).show(prim)

## sys_bob['true']['str'].show()
#. "'true'"
## sys_bob['true']['&'].show()
#. '{()}'
## core.Number(5)['succ'].show()
#. '6'
## core.Bob(core.String('yo')['++'], {'arg1': lambda a, b: core.String('zz')})['()'].show()
#. "'yozz'"
## run('"hey" ++ "dude"', prim=str)
#. 'heydude'

## run('sys.cons {first=5, rest=sys.empty}')
#. '(5:())'
## run('sys.cons{first=5, rest=sys.cons{first="hi", rest=sys.empty}}.length')
#. '2'

## run('sys.vector{elements = sys.cons {first=5, rest=sys.empty}}')
#. '[(5:())]'
## run('sys.vector{elements = sys.cons {first=5, rest=sys.empty}}.add_to(17)')
#. '[(22:())]'
## run('7 + sys.vector{elements = sys.cons {first=5, rest=sys.empty}}')
#. '[(12:())]'
