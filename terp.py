"""
Interpreter for a toy dialect of Bicicleta.
"""

from collections import namedtuple
from peglet import Parser, hug

class VarRef(namedtuple('_VarRef', 'name')):
    def __repr__(self):
        return '@' + self.name
    def eval(self, env):
        return env[self.name]

class NumberLiteral(namedtuple('_NumberLiteral', 'value')):
    def __repr__(self):
        return str(self.value)
    def eval(self, env):
        return Number(self.value)

class Call(namedtuple('_Call', 'receiver selector')):
    def __repr__(self):
        return '%s.%s' % (self.receiver, self.selector)
    def eval(self, env):
        return call(self.receiver.eval(env), self.selector)

class Extend(namedtuple('_Extend', 'base name bindings')):
    def __repr__(self):
        return '%s{%s: %s}' % (self.base, self.name,
                               ', '.join('%s=%s' % binding
                                         for binding in self.bindings))
    def eval(self, env):
        return extend(self.base.eval(env),
                      {slot: lambda rcvr: expr.eval(extend_env(env,
                                                               {self.name: rcvr}))
                       for slot, expr in self.bindings})

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
    return true_claim if value else false_claim

true_claim  = {'if': lambda _: {'()': lambda picking: call(picking, 'so')}}
false_claim = {'if': lambda _: {'()': lambda picking: call(picking, 'not')}}

def extend_env(env, bindings):
    result = dict(env)
    result.update(bindings)
    return result

initial_env = {'<>': {},
               'top': {'a': lambda rcvr: 5}} # XXX remove

def attach_affixes(primary, *affixes):
    expr = primary
    for affix in affixes:
        expr = attach(expr, affix)
    return expr

def attach(expr, affix): return affix[0](expr, *affix[1:])

def mk_int(string): return NumberLiteral(int(string))

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

## parse("{foo = {bar: x = y, '()'=x.a.b }}")
#. (@<>{: foo=@<>{bar: x=@y, ()=@x.a.b}},)

## bar, = parse("top.a")
## bar.eval(initial_env)
#. 5

## foo, = parse("{_: b={blah: x=top}}.b.x.a")
## foo
#. @<>{_: b=@<>{blah: x=@top}}.b.x.a
## foo.eval(initial_env)
#. 5

## numlit, = parse("137")
## numlit
#. 137

## adding, = parse("137.'+' {_=1}.'()'")
## adding
#. 137.+{: _=1}.()
## adding.eval(initial_env)['__value__']
#. 138

## cmping, = parse("137.'=='(_=1).if(so=42, not=168)")
## cmping == parse("137.'=='{_=1}.'()'.if{so=42, not=168}.'()'")[0]
#. True
## cmping.eval(initial_env)['__value__']
#. 168

fac, = parse("""
{env: 
 fac = {fac:
        '()' = fac.n.'=='(_=0).if(so  = 1,
                                  not = fac.n.'*'(_ = env.fac(n = fac.n.'-'(_=1))))}
}.fac(n=0)""")
## fac
#. @<>{env: fac=@<>{fac: ()=@fac.n.=={: _=0}.().if{: so=1, not=@fac.n.*{: _=@env.fac{: n=@fac.n.-{: _=1}.()}.()}.()}.()}}.fac{: n=0}.()
# fac.eval(initial_env)
