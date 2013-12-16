"""
Interpreter for a dialect of Bicicleta. I changed the 'if' syntax and
left out some things.
"""

from __future__ import division
from __future__ import print_function
from functools import reduce

from peglet import OneResult, Parser, hug


# Top level

def stepping(program):
    return run(program, loud=True)

def trampoline(state, loud=False):
    k, arg = state
    if loud:
        while k:
            whats_bouncing(k, arg)
            print('')
            k, arg = k[0](arg, *k)
    else:
        while k:
            k, arg = k[0](arg, *k)
    return arg

def whats_bouncing(k, arg):
    print(':', arg)
    while k:
        print(k[0].__name__, '\t', *k[1:-1])
        k = k[-1]

def run(program, loud=False):
    if isinstance(program, string_type):
        program = parse(program)
    return trampoline(program.eval(empty_env, (show_k, None)),
                      loud)

def show_k(result, _, k): return show(result, repr, k)


# Objects

def call(self, slot, k):
    if isinstance(self, Bob):
        value = self.get(slot)
        if value is not None:
            return k, value
        ancestor = self
        while True:
            method = ancestor.methods.get(slot)
            if method is not None:
                break
            if ancestor.parent is None:
                method = miranda_methods[slot]
                break
            ancestor = ancestor.parent
            if not isinstance(ancestor, Bob):
                method = primitive_method_tables[type(ancestor)].get(slot)
                if method is None: method = miranda_methods[slot]
                break
        return method(ancestor, self, (cache_slot_k, self, slot, k))
    else:
        method = primitive_method_tables[type(self)].get(slot)
        if method is None: method = miranda_methods[slot]
        return method(self, self, k)

def cache_slot_k(value, _, self, slot, k):
    self[slot] = value
    return k, value

def show(bob, prim, k):
    slot = 'repr' if prim is repr else 'str'
    return call(bob, slot, (show_slot_k, k))

class Bob(dict): # (Short for 'Bicicleta object'. But a lowercase bob might be a primitive instead.)
    parent = None
    def __init__(self, parent, methods):
        self.parent = parent
        self.methods = methods
    def __repr__(self):
        return trampoline(show(self, repr, None))

def list_slots(bob):
    ancestor, slots = bob, set()
    while isinstance(ancestor, Bob):
        slots.update(ancestor.methods)
        ancestor = ancestor.parent
    return slots

def show_slot_k(result, _, k):
    return k, (result if isinstance(result, string_type) else '<bob>')

class PrimCall(Bob):
    name = 'reflective slot value'
    def __init__(self, receiver):
        self.receiver = receiver
    methods = {
        '()': lambda self, doing, k: call(doing, 'arg1', (prim_call_k, self, k))
    }
def prim_call_k(arg1, _, self, k):
    assert isinstance(arg1, string_type), "Non-string slot: %r" % (arg1,)
    return call(self.receiver, arg1, k)

## run(""" 5{is_string=42}.'reflective slot value'("is_string") """)
#. '42'

miranda_methods = {
    'is_number': lambda ancestor, self, k: (k, True),
    'is_string': lambda ancestor, self, k: (k, False),
    'repr':      lambda ancestor, self, k: (k, miranda_show(ancestor, repr, self)),
    'str':       lambda ancestor, self, k: (k, miranda_show(ancestor, str, self)),
    PrimCall.name: lambda _, self, k:      (k, PrimCall(self)),
}

number_type = (int, float)
string_type = str               # XXX or unicode, in python2

def miranda_show(primval, prim_to_str, bob):
    shown = '' if isinstance(primval, Bob) else prim_to_str(primval)
    slots = list_slots(bob)
    if slots: shown += '{' + ', '.join(sorted(slots)) + '}' 
    return shown

class PrimOp(Bob):
    def __init__(self, ancestor, arg0):
        self.ancestor = ancestor
        self.arg0 = arg0


# Primitive objects

def prim_add(self, doing, k):
    return call(doing, 'arg1', (add_k, self, k))

def add_k(arg1, _, self, k):
    if isinstance(arg1, number_type):
        return k, self.ancestor + arg1
    else:
        return call(arg1, 'add_to', (add_to_k, self.arg0, k))

def add_to_k(arg1_add_to, _, arg0, k):
    return call(Bob(arg1_add_to, {'arg1': lambda _, __, mk: (mk, arg0)}),
                '()', k)

class PrimAdd(PrimOp):
    name, methods = '+', {
        '()': prim_add
    }

# The other arith ops should also do double dispatching, but for now here
# they are unconverted, since mainly I wanted to make sure it'd work, and
# I don't know what Kragen wants in detail.

class BarePrimOp(Bob):
    methods = {
        '()': lambda self, doing, k: call(doing, 'arg1', (self.arg1_k, self, k))
    }
    def __init__(self, ancestor, arg0):
        self.pv = ancestor

def sub_k(arg1, _, self, k): return k, self.pv - arg1
def mul_k(arg1, _, self, k): return k, self.pv * arg1
def div_k(arg1, _, self, k): return k, self.pv / arg1
def pow_k(arg1, _, self, k): return k, self.pv ** arg1
# XXX cmp ops need to deal with overriding:
def eq_k(arg1, _, self, k):  return k, self.pv == arg1
def lt_k(arg1, _, self, k):  return k, self.pv < arg1

class PrimSub(BarePrimOp): name, arg1_k = '-',  staticmethod(sub_k)
class PrimMul(BarePrimOp): name, arg1_k = '-',  staticmethod(mul_k)
class PrimDiv(BarePrimOp): name, arg1_k = '/',  staticmethod(div_k)
class PrimPow(BarePrimOp): name, arg1_k = '**', staticmethod(pow_k)
class PrimEq(BarePrimOp):  name, arg1_k = '==', staticmethod(eq_k)
class PrimLt(BarePrimOp):  name, arg1_k = '<',  staticmethod(lt_k)

def primop_method(class_):
    return lambda ancestor, receiver, k: (k, class_(ancestor, receiver))

number_methods = {
    'is_number': lambda _, me, k: (k, True),
    '+':         primop_method(PrimAdd),
    '-':         primop_method(PrimSub),
    '*':         primop_method(PrimMul),
    '/':         primop_method(PrimDiv),
    '**':        primop_method(PrimPow),
    '==':        primop_method(PrimEq),
    '<':         primop_method(PrimLt),
}

def string_cat_k(arg1, _, self, k):
    assert isinstance(arg1, string_type), arg1
    return k, self.pv + arg1
class StringCat(BarePrimOp): name, arg1_k = '++', staticmethod(string_cat_k)

string_methods = {
    'is_string': lambda _, me, k: (k, True),
    'is_empty':  lambda ancestor, me, k: (k, ancestor == ''),
    'first':     lambda ancestor, me, k: (k, ancestor[0]),
    'rest':      lambda ancestor, me, k: (k, ancestor[1:]),
    '==':        primop_method(PrimEq),
    '<':         primop_method(PrimLt),
    '++':        primop_method(StringCat),
}

bool_methods = {
    'if':   lambda _, me, k: (k, (pick_so if me else pick_else)),
}
pick_so     = Bob(None, {'()': lambda _, doing, k: call(doing, 'so', k)})
pick_else   = Bob(None, {'()': lambda _, doing, k: call(doing, 'else', k)})

primitive_method_tables = {
    bool:  bool_methods,
    int:   number_methods,
    float: number_methods,
    str:   string_methods,
}

root_bob = Bob(None, {})


# Evaluation

class VarRef(object):
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return self.name
    def eval(self, env, k):
        return k, env[self.name]

class Literal(object):
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return repr(self.value)
    def eval(self, env, k):
        return k, self.value

class Call(object):
    def __init__(self, receiver, slot):
        self.receiver = receiver
        self.slot = slot
    def __repr__(self):
        return '%s.%s' % (self.receiver, self.slot)
    def eval(self, env, k):
        return self.receiver.eval(env, (call_k, self.slot, k))

def call_k(bob, _, slot, k):
    return call(bob, slot, k)

def make_extend(base, name, bindings):
    extend = SelflessExtend if name is None else Extend
    # (We needn't special-case this; it's an optimization.)
    return extend(base, name, bindings)

class Extend(object):
    def __init__(self, base, name, bindings):
        self.base = base
        self.name = name
        self.bindings = bindings
    def __repr__(self):
        return '%s{%s%s}' % (self.base,
                             self.name + ': ' if self.name else '',
                               ', '.join('%s=%s' % binding
                                         for binding in self.bindings))
    def eval(self, env, k):
        return self.base.eval(env, (extend_k, self, env, k))

def extend_k(bob, _, self, env, k):
    return k, Bob(bob,
                  {slot: make_slot_thunk(self.name, expr, env)
                   for slot, expr in self.bindings})

class SelflessExtend(Extend):
    def eval(self, env, k):
        return self.base.eval(env, (selfless_extend_k, self, env, k))

def selfless_extend_k(bob, _, self, env, k):
    return k, Bob(bob,
                  {slot: make_selfless_slot_thunk(expr, env)
                   for slot, expr in self.bindings})

def make_selfless_slot_thunk(expr, env):
    return lambda _, __, k: expr.eval(env, k)

def make_slot_thunk(name, expr, env):
    def thunk(_, receiver, k):
        new_env = dict(env)
        new_env[name] = receiver
        return expr.eval(new_env, k)
    return thunk

empty_env = {}


# Parser

program_grammar = r"""
program     = expr _ !.
expr        = factor infixes                attach_all
factor      = primary affixes               attach_all

primary     = name                          VarRef
            | _ (\d*\.\d+)                  float Literal
            | _ (\d+)                       int   Literal
            | _ "([^"\\]*)"                       Literal
            | _ \( _ expr \)
            | empty derive                  attach

affixes     = affix affixes | 
affix       = _ [.] name                    defer_dot
            | derive
            | _ \( bindings _ \)            defer_funcall
            | _ \[ bindings _ \]            defer_squarecall

derive      = _ { name _ : bindings _ }     defer_derive
            | _ { nameless bindings _ }     defer_derive
bindings    = binds                         name_positions
binds       = binding newline binds
            | binding _ , binds
            | binding
            | 
binding     = name _ [=] expr               hug
            | positional expr               hug

infixes     = infix infixes | 
infix       = infix_op factor               defer_infix
infix_op    = _ !lone_eq opchars
opchars     = ([-~`!@$%^&*+<>?/|\\=]+)
lone_eq     = [=] !opchars

name        = _ ([A-Za-z_][A-Za-z_0-9]*)
            | _ '([^'\\]*)'

newline     = blanks \n
blanks      = blank blanks | 
blank       = !\n (?:\s|#.*)

_           = (?:\s|#.*)*
"""
# TODO: support backslashes in '' and ""
# TODO: foo(name: x=y) [if actually wanted]

empty_literal = Literal(root_bob)

def empty(): return empty_literal
def nameless(): return None
def positional(): return None

def name_positions(*bindings):
    return tuple((('arg%d' % i if slot is None else slot), expr)
                 for i, (slot, expr) in enumerate(bindings, 1))

def attach_all(expr, *affixes):    return reduce(attach, affixes, expr)
def attach(expr, affix):           return affix[0](expr, *affix[1:])

def defer_dot(name):               return Call, name
def defer_derive(name, bindings):  return make_extend, name, bindings
def defer_funcall(bindings):       return mk_funcall, '()', bindings
def defer_squarecall(bindings):    return mk_funcall, '[]', bindings
def defer_infix(operator, expr):   return mk_infix, operator, expr

def mk_funcall(expr, slot, bindings):
    "  foo(x=y) ==> foo{x=y}.'()'  "
    return Call(make_extend(expr, nameless(), bindings), slot)

def mk_infix(left, operator, right):
    "   x + y ==> x.'+'(_=y)  "
    return mk_funcall(Call(left, operator), '()', (('arg1', right),))

parse = OneResult(Parser(program_grammar, int=int, float=float, **globals()))


# Crude tests and benchmarks

## parse("x ++ y{a=b} <*> z.foo")
#. x.++{arg1=y{a=b}}.().<*>{arg1=z.foo}.()

## parse('5')
#. 5

## run('5')
#. '5'

## wtf = parse("{x=42, y=55}.x")
## wtf
#. {x=42, y=55}.x
## run(wtf)
#. '42'

## run("{y=42, x=55, z=137}.x")
#. '55'

## parse("137")
#. 137
## parse("137[yo=dude]")
#. 137{yo=dude}.[]

## adding = parse("137.'+' {arg1=1}.'()'")
## adding
#. 137.+{arg1=1}.()
## run(adding)
#. '138'

## run('3+2')
#. '5'
## run('3*2')
#. '6'

## run("137.5 - 2 - 1")
#. '134.5'

## run("(136 < 137).if(so=1, else=2)")
#. '1'
## run("(137 < 137).if(so=1, else=2)")
#. '2'
## run("137.'<' {arg1=137}.'()'.if(so=1, else=2)")
#. '2'

## cmping = parse("(137 == 1).if(so=42, else=168)")
## repr(cmping) == repr(parse("137.'=='{arg1=1}.'()'.if{so=42, else=168}.'()'"))
#. True
## run(cmping)
#. '168'

## run('"howdy"')
#. "'howdy'"
## run('("hello" == "aloha").if(so=42, else=168)')
#. '168'
## run('("hello" == "hello").if(so=42, else=168)')
#. '42'

test_extend = parse("""
    {main:
     three = {me: x = 3, xx = me.x + me.x},
     four = main.three{x=4},
     result = main.three.xx + main.four.xx
    }.result
""")
## run(test_extend)
#. '14'

## run('"hey " ++ 42.str ++ " and " ++ (1136+1).str.rest')
#. "'hey 42 and 137'"
# run('"hey {x} and {why}" % {x=84/2, why=136+1}')
## run("5**3")
#. '125'

## run("5{}*6")
#. '30'

## run("5.is_string")
#. 'False'
## run("5.is_number")
#. 'True'

def make_fac(n):
    fac = parse("""
{env: 
 fac = {fac:   # fac for factorial
        '()' = (fac.n == 0).if(so = 1,
                               else = fac.n * env.fac(n = fac.n-1))}
}.fac(n=%d)""" % n)
    return fac

fac = make_fac(4)
## fac
#. {env: fac={fac: ()=fac.n.=={arg1=0}.().if{so=1, else=fac.n.*{arg1=env.fac{n=fac.n.-{arg1=1}.()}.()}.()}.()}}.fac{n=4}.()
## run(fac)
#. '24'

def make_fib(n):
    fib = parse("""
{env:
 fib = {fib:
        '()' = (fib.n < 2).if(so = 1,
                              else = env.fib(n=fib.n-1) + env.fib(n=fib.n-2))}
}.fib(n=%d)
    """ % n)
    return fib

## run(make_fib(5))
#. '8'
