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

sys_bob = core.Bob(None, {
    'true':  lambda _, me, k: (k, True),
    'false': lambda _, me, k: (k, False),
})
global_env = (sys_bob,)

def run(program, loud=False):
    if isinstance(program, core.string_type):
        program = core.parse(program)
    program.analyze({'sys': 0})
    return core.trampoline(program.eval(global_env, None),
                           loud)

def extend_in_place(methods, overlay):
    # TODO: deep copy? Shallow is all we need for now.
    for slot in overlay.methods:
        assert slot not in methods, slot
    methods.update(overlay.methods)

def load(filename):
    expr = core.parse(open(filename).read())
    expr.analyze({'sys': 0})
    return core.trampoline(expr.eval(global_env, None))

extend_in_place(core.bool_methods,   load('sys_bool.bicicleta'))
extend_in_place(core.number_methods, load('sys_number.bicicleta'))
extend_in_place(core.string_methods, load('sys_string.bicicleta'))
extend_in_place(sys_bob.methods,     load('sys.bicicleta'))

## run('5')
#. 5
## run('5+6')
#. 11

## run('"hey" ++ "dude"')
#. 'heydude'

## run('"" % {x=84/2, why=136+1}')
#. ''
## run('"abc" % {x=84/2, why=136+1}')
#. 'abc'
## run(""" "{}" % {''=5} """)
#. '5'

## run('"hey {x} and {why}" % {x=84/2, why=136+1}')
#. 'hey 42.0 and 137'

## run('sys.cons {first=5, rest=sys.empty}')
#. (5:())
## run('sys.cons{first=5, rest=sys.cons{first="hi", rest=sys.empty}}.length')
#. 2

## run('sys.vector{elements = sys.empty}')
#. [()]
## run('sys.vector{elements = sys.cons {first=5, rest=sys.empty}}')
#. [(5:())]
## run('sys.vector{elements = sys.cons {first=5, rest=sys.empty}}.add_to(17)')
#. [(22:())]
## run('7 + sys.vector{elements = sys.cons {first=5, rest=sys.empty}}')
#. [(12:())]
