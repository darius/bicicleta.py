"""
Interpreter for a toy dialect of Bicicleta.
"""

from collections import namedtuple
from peglet import Parser, hug

class VarRef(object):
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return '@' + self.name
    def eval(self, env):
        return env[self.name]

class NumberLiteral(object):
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return str(self.value)
    def eval(self, env):
        return Number(self.value)

class Call(object):
    def __init__(self, receiver, selector):
        self.receiver = receiver
        self.selector = selector
    def __repr__(self):
        return '%s.%s' % (self.receiver, self.selector)
    def eval(self, env):
        return call(self.receiver.eval(env), self.selector)

class Extend(object):
    def __init__(self, base, name, bindings):
        self.base = base
        self.name = name
        self.bindings = list(bindings)
    def __repr__(self):
        return '%s{%s: %s}' % (self.base, self.name,
                               ', '.join('%s=%s' % binding
                                         for binding in self.bindings))
    def eval(self, env):
        return Thunk(self, env)
    def force(self, env):
        return extend(
            self.base.eval(env),
            {slot: (lambda expr:
                    lambda rcvr: expr.eval(extend_env(env, {self.name: rcvr}))
                )(expr)
             for slot, expr in self.bindings})

class Thunk(object): 
    def __init__(self, expr, env):
        assert isinstance(expr, Extend)
        self.expr = expr
        self.env = env
    def __repr__(self):
        return 'Thunk(%r)' % self.expr
    def force(self):
        return self.expr.force(self.env)
    def get(self, key):
        return self.force().get(key)
    def __getitem__(self, key):
        return self.force().__getitem__(key)
    def items(self):
        return self.force().items()
    def __iter__(self):
        return iter(self.force().items())

def call(receiver, selector):
    return receiver[selector](receiver)

def extend(thing, bindings):
    return extend_env(thing, bindings)

def Number(n):
    return {'__value__': n,
            '+': lambda _: {'()': lambda doing:
                            Number(n + call(doing, '_')['__value__'])},
            '-': lambda _: {'()': lambda doing:
                            Number(n - call(doing, '_')['__value__'])},
            '*': lambda _: {'()': lambda doing:
                            Number(n * call(doing, '_')['__value__'])},
            '==': lambda _: {'()': lambda operation:
                             Claim(n == call(operation, '_').get('__value__'))}}

def Claim(value):
    assert isinstance(value, bool)
    return true_claim if value else false_claim

true_claim  = {'if': lambda _: {'()': lambda picking: call(picking, 'so')}}
false_claim = {'if': lambda _: {'()': lambda picking: call(picking, 'not')}}

def extend_env(env, bindings):
    result = dict(env)
    result.update(bindings)
    return result

initial_env = {'<>': {}}

def attach_affixes(primary, *affixes):
    expr = primary
    for affix in affixes:
        expr = attach(expr, affix)
    return expr

def attach(expr, affix): return affix[0](expr, *affix[1:])

def mk_int(wtf):
    return NumberLiteral(int(wtf))

def empty(): return VarRef('<>')
def nameless(): return ''

def defer_dot(name):               return Call, name
def defer_derive(name, *bindings): return Extend, name, bindings
def defer_funcall(*bindings):      return mk_funcall, bindings

def mk_funcall(expr, bindings):
    "  foo(x=y) ==> foo{x=y}.'()'  "
    return Call(Extend(expr, nameless(), bindings), '()')

# TODO: compare yacc grammar

parse = Parser(r"""
program     = _ expr !.
expr        = primary affixes               attach_affixes
primary     = name                          VarRef
            | (\d+) _                       mk_int
            | \( _ expr \) _
            | empty derive                  attach
affixes     = affix affixes | 
affix       = [.] name                      defer_dot
            | derive
            | \( _ bindings \) _            defer_funcall
derive      = { _ name : _ bindings } _     defer_derive
            | { _ nameless bindings } _     defer_derive
bindings    = binding , _ bindings
            | binding
            | 
binding     = name [=] _ expr               hug
name        = ([A-Za-z_][A-Za-z_0-9]*) _
            | '([^']*)' _
_           = \s*
""", **globals())

# XXX backslashes in quoted names, blah lblah

## parse('x = y', rule='bindings')
#. (('x', @y),)

## wtf, = parse("{x=42, y=55}.x")
## wtf
#. @<>{: x=42, y=55}.x
## wtf.eval(initial_env)['__value__']
#. 42

## parse("{y=42, x=55, z=137}.x")[0].eval(initial_env)['__value__']
#. 55

## numlit, = parse("137")
## numlit
#. 137

## adding, = parse("137.'+' {_=1}.'()'")
## adding
#. 137.+{: _=1}.()
## adding.eval(initial_env)['__value__']
#. 138

## cmping, = parse("137.'=='(_=1).if(so=42, not=168)")
## repr(cmping) == repr(parse("137.'=='{_=1}.'()'.if{so=42, not=168}.'()'")[0])
#. True
## cmping.eval(initial_env)['__value__']
#. 168

fac, = parse("""
{env: 
 fac = {fac:
        '()' = fac.n.'=='(_=0).if(so  = 1,
                                  not = fac.n.'*'(_ = env.fac(n = fac.n.'-'(_=1))))}
}.fac(n=4)""")
## fac
#. @<>{env: fac=@<>{fac: ()=@fac.n.=={: _=0}.().if{: so=1, not=@fac.n.*{: _=@env.fac{: n=@fac.n.-{: _=1}.()}.()}.()}.()}}.fac{: n=4}.()
## fac.eval(initial_env)['__value__']
#. 24
