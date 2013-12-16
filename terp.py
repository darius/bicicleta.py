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
    bob = core.trampoline(core.call(sys_bob, name, None))
    extend_in_place(bob, load('sys_%s.bicicleta' % name))

def extend_in_place(bob, overlay):
    # TODO: deep copy? Shallow is all we need for now.
    for slot, method in overlay.methods.items():
        assert slot not in bob
        assert slot not in bob.methods, (slot, bob)
        bob.methods[slot] = method

def load(filename):
    expr = core.parse(open(filename).read())
    return core.trampoline(expr.eval(global_env, None))

sys_bob = core.Prim(None, {
    'true':  lambda _, me, k: (k, core.true_claim),
    'false': lambda _, me, k: (k, core.false_claim),
})
global_env = {'sys': sys_bob}

sys_load('true')
sys_load('false')
extend_in_place(core.Number(42), load('sys_number.bicicleta'))
extend_in_place(core.String('hi'), load('sys_string.bicicleta'))
extend_in_place(sys_bob, load('sys.bicicleta'))

def run(text):
    program = core.parse(text)
    return core.trampoline(program.eval(global_env, (core.show_k, None)))

def debug():
    text = """ "{}" % {''=5} """
    program = core.parse(text)
    core.trampoline(program.eval(global_env, None),
                    loud=True)

## run('5')
#. '5'
## run('5+6')
#. '11'

## run('"hey" ++ "dude"')
#. "'heydude'"

## run('"" % {x=84/2, why=136+1}')
#. "''"
## run('"abc" % {x=84/2, why=136+1}')
#. "'abc'"
## run(""" "{}" % {''=5} """)
#. "'5'"

## run('"hey {x} and {why}" % {x=84/2, why=136+1}')
#. "'hey 42.0 and 137'"

## run('sys.cons {first=5, rest=sys.empty}')
#. '(5:())'
## run('sys.cons{first=5, rest=sys.cons{first="hi", rest=sys.empty}}.length')
#. '2'

## run('sys.vector{elements = sys.empty}')
#. '[()]'
## run('sys.vector{elements = sys.cons {first=5, rest=sys.empty}}')
#. '[(5:())]'
## run('sys.vector{elements = sys.cons {first=5, rest=sys.empty}}.add_to(17)')
#. '[(22:())]'
## run('7 + sys.vector{elements = sys.cons {first=5, rest=sys.empty}}')
#. '[(12:())]'
