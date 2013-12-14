"""
Interpreter for a dialect of Bicicleta. I changed the 'if' syntax and
left out some things.
"""

from __future__ import division
import sys; sys.setrecursionlimit(7500)
from functools import reduce

from peglet import OneResult, Parser, hug


# Top level

def run(program):
    if isinstance(program, str):
        program = parse(program)
    return program.eval(empty_env).show()


# Objects

class Bob(dict):    # (short for 'Bicicleta object')
    parent = None
    primval = None
    def __init__(self, parent, methods):
        self.parent = parent
        self.methods = methods
    def __missing__(self, slot):
        ancestor = self
        while True:
            try:
                method = ancestor.methods[slot]
            except KeyError:
                ancestor = ancestor.parent
                if ancestor is None:
                    raise
            else:
                value = self[slot] = method(ancestor, self)
                return value
    def show(self, prim=repr):
        return '{%s}' % ', '.join(sorted(self.list_slots()))
    def list_slots(self):
        ancestor, slots = self, set()
        while ancestor is not None:
            slots.update(ancestor.methods)
            ancestor = ancestor.parent
        return slots

class Prim(Bob):
    def __init__(self, primval, methods):
        self.primval = primval
        self.methods = methods
    def show(self, prim=repr):
        return prim(self.primval)

class BarePrimOp(Bob):
    def __init__(self, ancestor, arg0):
        self.pv = ancestor.primval
    def show(self, prim=repr):
        return "%r.'%s'" % (self.pv, self.op)


# Primitive objects

class PrimAdd(BarePrimOp):
    name, methods = '+', {
        '()': lambda self, doing: Number(self.pv + doing['arg1'].primval)
    }
class PrimSub(BarePrimOp):
    name, methods = '-', {
        '()': lambda self, doing: Number(self.pv - doing['arg1'].primval)
    }
class PrimMul(BarePrimOp):
    name, methods = '*', {
        '()': lambda self, doing: Number(self.pv * doing['arg1'].primval)
    }
class PrimDiv(BarePrimOp):
    name, methods = '/', {
        '()': lambda self, doing: Number(self.pv / doing['arg1'].primval)
    }
class PrimPow(BarePrimOp):
    name, methods = '**', {
        '()': lambda self, doing: Number(self.pv ** doing['arg1'].primval)
    }

class PrimEq(BarePrimOp):     # XXX cmp ops need to deal with overriding
    name, methods = '==', {
        '()': lambda self, doing: Claim(self.pv == doing['arg1'].primval)
    }
class PrimLt(BarePrimOp):
    name, methods = '<', {
        '()': lambda self, doing: Claim(self.pv < doing['arg1'].primval)
    }

class Number(Prim):
    def __init__(self, n):
        self.primval = n
    methods = {
        '+':  PrimAdd,
        '-':  PrimSub,
        '*':  PrimMul,
        '/':  PrimDiv,
        '**': PrimPow,
        '==': PrimEq,
        '<':  PrimLt,
    }

class PrimStringSubst(BarePrimOp):
    name, methods = '%', {
        '()': lambda self, doing: String(string_substitute(self.pv,
                                                           doing['arg1']))
    }
def string_substitute(template, bob):
    import re
    return re.sub(r'{(.*?)}', lambda m: bob[m.group(1)].show(str),
                  template)

class String(Prim):
    def __init__(self, s):
        self.primval = s
    methods = {
        '==': PrimEq,
        '<':  PrimLt,
        '%':  PrimStringSubst,
    }

def Claim(value):
    assert isinstance(value, bool)
    return true_claim if value else false_claim

true_claim  = Prim(None, {'if': lambda _, me: pick_so})
false_claim = Prim(None, {'if': lambda _, me: pick_else})
pick_so     = Prim(None, {'()': lambda _, doing: doing['so']})
pick_else   = Prim(None, {'()': lambda _, doing: doing['else']})

root_bob = Prim(None, {})


# Evaluation

class VarRef(object):
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return self.name
    def eval(self, env):
        return env[self.name]

class Literal(object):
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return self.value.show()
    def eval(self, env):
        return self.value

class Call(object):
    def __init__(self, receiver, slot):
        self.receiver = receiver
        self.slot = slot
    def __repr__(self):
        return '%s.%s' % (self.receiver, self.slot)
    def eval(self, env):
        return self.receiver.eval(env)[self.slot]

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
    def eval(self, env):
        return Bob(self.base.eval(env),
                   {slot: make_slot_thunk(self.name, expr, env)
                    for slot, expr in self.bindings})

class SelflessExtend(Extend):
    def eval(self, env):
        return Bob(self.base.eval(env),
                   {slot: make_selfless_slot_thunk(expr, env)
                    for slot, expr in self.bindings})

def make_selfless_slot_thunk(expr, env):
    return lambda _, __: expr.eval(env)

def make_slot_thunk(name, expr, env):
    def thunk(_, receiver):
        new_env = dict(env)
        new_env[name] = receiver
        return expr.eval(new_env)
    return thunk

empty_env = {}


# Parser

program_grammar = r"""
program     = expr _ !.
expr        = factor infixes                attach_all
factor      = primary affixes               attach_all

primary     = name                          VarRef
            | _ (\d*\.\d+)                  float Number Literal
            | _ (\d+)                       int   Number Literal
            | _ "([^"\\]*)"                       String Literal
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

## wtf = parse("{x=42, y=55}.x")
## wtf
#. None{x=42, y=55}.x
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

## run('"hey {x} and {why}" % {x=84/2, why=136+1}')
#. "'hey 42.0 and 137'"
## run("5**3")
#. '125'

## run("5{}*6")
#. '30'

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
#. None{env: fac=None{fac: ()=fac.n.=={arg1=0}.().if{so=1, else=fac.n.*{arg1=env.fac{n=fac.n.-{arg1=1}.()}.()}.()}.()}}.fac{n=4}.()
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
