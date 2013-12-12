"""
Interpreter for a dialect of Bicicleta. I changed the 'if' syntax and
left out some things.
"""

import sys; sys.setrecursionlimit(2500)

from peglet import Parser, hug


# Top level

def run(program):
    if isinstance(program, (str, unicode)):
        program, = parse(program)
    return show(program.eval(initial_env))

def show(obj):
    if isinstance(obj, Thunk):
        return obj.show()
    value = obj.get('__value__')
    if value is not None:
        return repr(value)
    else:
        return '{%s}' % ', '.join(name for name, call_thunk in obj.items())


# Expressions, environments, and objects

class VarRef(object):
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return '@' + self.name
    def eval(self, env):
        return env[self.name]

class Literal(object):
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return show(self.value)
    def eval(self, env):
        return self.value

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
        return extend(self.base.eval(env),
                      {slot: make_slot_thunk(self.name, expr, env)
                       for slot, expr in self.bindings})

def make_slot_thunk(name, expr, env):
    return lambda rcvr: expr.eval(extend_env(env, {name: rcvr}))

class Thunk(object): 
    def __init__(self, expr, env):
        assert isinstance(expr, Extend)
        self.expr = expr
        self.env = env
        self.forced = None
    def __repr__(self):
        return 'Thunk(%r)' % self.expr
    def show(self):
        return show(self.forced) if self.forced else '$'
    def force(self):
        if self.forced is None:
            self.forced = self.expr.force(self.env)
        return self.forced
    def get(self, key):              return self.force().get(key)
    def __getitem__(self, key):      return self.force().__getitem__(key)
    def __setitem__(self, key, val): return self.force().__setitem__(key, val)
    def items(self):                 return self.force().items()
    def __iter__(self):              return iter(self.force().items())

def call(receiver, selector):
    what = receiver[selector]
    if callable(what):
        value = what(receiver)
        receiver[selector] = value
    else:
        value = what
    return value

def extend(thing, bindings):
    return extend_env(thing, bindings)

def extend_env(env, bindings):
    result = dict(env)
    result.update(bindings)
    return result

initial_env = {'<>': {}}


# Primitive objects

def Number(n):
    return {'__value__': n,
            '+':  lambda _: {'()': lambda doing:
                             Number(n + call(doing, 'arg1')['__value__'])},
            '-':  lambda _: {'()': lambda doing:
                             Number(n - call(doing, 'arg1')['__value__'])},
            '*':  lambda _: {'()': lambda doing:
                             Number(n * call(doing, 'arg1')['__value__'])},
            '==': lambda _: {'()': lambda operation:
                             Claim(n == call(operation, 'arg1').get('__value__'))},
            '<':  lambda _: {'()': lambda operation: # XXX should cmp of num and string be an error?
                             (lambda other: Claim(other.get('__value__') is not None
                                                  and n < other['__value__']))(call(operation, 'arg1'))}}

def String(s):
    return {'__value__': s,
            '==': lambda _: {'()': lambda operation:
                             Claim(s == call(operation, 'arg1').get('__value__'))},
            '<':  lambda _: {'()': lambda operation:
                             (lambda other: Claim(other.get('__value__') is not None
                                                  and s < other['__value__']))(call(operation, 'arg1'))}}

def Claim(value):
    assert isinstance(value, bool)
    return true_claim if value else false_claim

true_claim  = {'if': lambda _: {'()': lambda picking: call(picking, 'so')}}
false_claim = {'if': lambda _: {'()': lambda picking: call(picking, 'not')}}


# Parser

program_grammar = r"""
program     = _ expr !.
expr        = factor infixes                attach_all
factor      = primary affixes               attach_all

primary     = name                          VarRef
            | (\d*\.\d+) _                  float Number Literal
            | (\d+) _                       int   Number Literal
            | "([^"\\]*)" _                       String Literal
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

infixes     = infix infixes | 
infix       = infix_op factor               defer_infix
infix_op    = !lone_eq opchars _
opchars     = ([-~`!@$%^&*+<>?/|\\=]+)
lone_eq     = [=] !opchars

name        = ([A-Za-z_][A-Za-z_0-9]*) _
            | '([^'\\]*)' _
_           = (?:\s|#.*)*
"""
# TODO: support backslashes in '' and ""
# TODO: comma optionally a newline instead
# TODO: foo(name: x=y) [if actually wanted]
# TODO: foo[]
# TODO: positional arguments

def empty(): return VarRef('<>')
def nameless(): return ''

def attach_all(expr, *affixes):    return reduce(attach, affixes, expr)
def attach(expr, affix):           return affix[0](expr, *affix[1:])

def defer_dot(name):               return Call, name
def defer_derive(name, *bindings): return Extend, name, bindings
def defer_funcall(*bindings):      return mk_funcall, bindings
def defer_infix(operator, expr):   return mk_infix, operator, expr

def mk_funcall(expr, bindings):
    "  foo(x=y) ==> foo{x=y}.'()'  "
    return Call(Extend(expr, nameless(), bindings), '()')

def mk_infix(left, operator, right):
    "   x + y ==> x.'+'(_=y)  "
    return mk_funcall(Call(left, operator), (('arg1', right),))

parse = Parser(program_grammar, int=int, float=float, **globals())


# Crude tests and benchmarks

## parse("x ++ y{a=b} <*> z.foo")
#. (@x.++{: arg1=@y{: a=@b}}.().<*>{: arg1=@z.foo}.(),)

## wtf, = parse("{x=42, y=55}.x")
## wtf
#. @<>{: x=42, y=55}.x
## run(wtf)
#. '42'

## parse("{y=42, x=55, z=137}.x")[0].eval(initial_env)['__value__']
#. 55

## numlit, = parse("137")
## numlit
#. 137

## adding, = parse("137.'+' {arg1=1}.'()'")
## adding
#. 137.+{: arg1=1}.()
## run(adding)
#. '138'

## run("137.5 - 2 - 1")   # N.B. associates right to left, currently
#. '134.5'

## run("(136 < 137).if(so=1, not=2)")
#. '1'
## run("(137 < 137).if(so=1, not=2)")
#. '2'
## run("137.'<' {arg1=137}.'()'.if(so=1, not=2)")
#. '2'

## cmping, = parse("(137 == 1).if(so=42, not=168)")
## repr(cmping) == repr(parse("137.'=='{arg1=1}.'()'.if{so=42, not=168}.'()'")[0])
#. True
## run(cmping)
#. '168'

## run('"howdy"')
#. "'howdy'"
## run('("hello" == "aloha").if(so=42, not=168)')
#. '168'
## run('("hello" == "hello").if(so=42, not=168)')
#. '42'

def make_fac(n):
    fac, = parse("""
{env: 
 fac = {fac:   # fac for factorial
        '()' = (fac.n == 0).if(so  = 1,
                               not = fac.n * env.fac(n = fac.n-1))}
}.fac(n=%d)""" % n)
    return fac

fac = make_fac(4)
## fac
#. @<>{env: fac=@<>{fac: ()=@fac.n.=={: arg1=0}.().if{: so=1, not=@fac.n.*{: arg1=@env.fac{: n=@fac.n.-{: arg1=1}.()}.()}.()}.()}}.fac{: n=4}.()
## run(fac)
#. '24'

def make_fib(n):
    fib, = parse("""
{env:
 fib = {fib:
        '()' = (fib.n < 2).if(so = 1,
                              not = env.fib(n=fib.n-1) + env.fib(n=fib.n-2))}
}.fib(n=%d)
    """ % n)
    return fib

## run(make_fib(5))
#. '8'

def make_tarai():
    # TARAI is like TAK, but it's much faster with lazy evaluation.
    # It was Takeuchi's original function.
    program, = parse("""
{env:
 tarai = {tarai: 
          '()' = (tarai.y < (tarai.x)).if(
              so = env.tarai(x=env.tarai(x=tarai.x-1, y=tarai.y, z=tarai.z),
                             y=env.tarai(x=tarai.y-1, y=tarai.z, z=tarai.x),
                             z=env.tarai(x=tarai.z-1, y=tarai.x, z=tarai.y)),
              not = tarai.y)
         }
    }.tarai(x=18, y=12, z=6)""")
    return program

def make_tak():
    program, = parse("""
{env:
 tak = {tak: 
          '()' = (tak.y < (tak.x)).if(
              so = env.tak(x=env.tak(x=tak.x-1, y=tak.y, z=tak.z),
                           y=env.tak(x=tak.y-1, y=tak.z, z=tak.x),
                           z=env.tak(x=tak.z-1, y=tak.x, z=tak.y)),
              not = tak.z)
         }
    }.tak(x=18, y=12, z=6)""")
    return program

test_extend, = parse("""
    {main:
     three = {x = 3},
     four = main.three{x=4},
     seven = main.three.x + main.four.x
    }.seven
""")
## run(test_extend)
#. '7'

def timed(f):
    import time
    start = time.clock()
    result = f()
    return time.clock() - start, result

def bench(bound=15):
    print '%5s  %3s %13s' % ('Secs', 'N', 'fac N')
    for n in range(bound):
        fac = make_fac(n)
        seconds, result = timed(lambda: fac.eval(initial_env)['__value__'])
        print '%5.3g  %3d %13d' % (seconds, n, result)

def bench2():
    tarai = make_tarai()
    print timed(lambda: run(tarai))

def bench3():
    tak = make_tak()
    print timed(lambda: run(tak))

itersum3, = parse("""
{env:
 outer = {outer: 
   i=0, sum=0,
   '()' = (outer.i == 0).if(
       so = outer.sum,
       not = env.outer(i=outer.i-1, sum=outer.sum + env.mid(i=outer.i)))},
 mid = {mid: 
   i=0, sum=0,
   '()' = (mid.i == 0).if(
       so = mid.sum,
       not = env.mid(i=mid.i-1, sum=mid.sum + env.inner(i=mid.i)))},
 inner = {inner:
   i=0, sum=0,
   '()' = (inner.i == 0).if(
       so = inner.sum,
       not = env.inner(i=inner.i-1, sum=inner.sum + inner.i))},
 main = env.outer(i = 40)
}.main
""")

if __name__ == '__main__':
    print timed(lambda: run(itersum3))
    fib = make_fib(20)
    print timed(lambda: run(fib))
    print bench3()
