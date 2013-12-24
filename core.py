"""
Interpreter for a dialect of Bicicleta. I changed the 'if' syntax and
left out some things.
"""

from __future__ import division
from __future__ import print_function
from functools import reduce

from peglet import OneResult, Parser, hug


# The trampoline reifies execution state to avoid Python stack
# overflow and make a debugging UI possible.

def trampoline(state, trace=False):
    k, value = state
    if trace:
        while k:
            whats_bouncing(k, value)
            print('')
            fn, free_var, k = k
            k, value = fn(value, free_var, k)
    else:
        while k:
            fn, free_var, k = k
            k, value = fn(value, free_var, k)
    return value

def whats_bouncing(k, value):
    print(':', value)
    while k:
        print(k[0].__name__, '\t', *k[1:-1])
        k = k[-1]


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
        return method(ancestor, self, (cache_slot_k, (self, slot), k))
    else:
        method = primitive_method_tables[type(self)].get(slot)
        if method is None: method = miranda_methods[slot]
        return method(self, self, k)

def cache_slot_k(value, free_var, k):
    self, slot = free_var
    self[slot] = value
    return k, value

class Bob(dict): # (Short for 'Bicicleta object'. But a lowercase bob might be a primitive instead.)
    parent = None
    def __init__(self, parent, methods):
        self.parent = parent
        self.methods = methods
    def __repr__(self):
        return trampoline(show(self, 'repr', None))
    def __str__(self):
        return trampoline(show(self, 'str', None))

def show(bob, slot, k):
    return call(bob, slot, (show_slot_k, None, k))

def show_slot_k(result, _, k):
    return k, (result if isinstance(result, string_type) else '<bob>')

def list_slots(bob):
    ancestor, slots = bob, set()
    while isinstance(ancestor, Bob):
        slots.update(ancestor.methods)
        ancestor = ancestor.parent
    return slots


# Miranda methods: any you don't define yourself are supplied for you.

class PrimCall(Bob):
    name = 'reflective slot value'
    def __init__(self, receiver):
        self.receiver = receiver
    methods = {
        '()': lambda self, doing, k: call(doing, 'arg1', (prim_call_k, self, k))
    }
def prim_call_k(arg1, self, k):
    assert isinstance(arg1, string_type), "Non-string slot: %r" % (arg1,)
    return call(self.receiver, arg1, k)

miranda_methods = {
    'is_number': lambda ancestor, self, k: (k, False),
    'is_string': lambda ancestor, self, k: (k, False),
    'repr':      lambda ancestor, self, k: (k, miranda_show(ancestor, repr, self)),
    'str':       lambda ancestor, self, k: (k, miranda_show(ancestor, str, self)),
    PrimCall.name: lambda _, self, k:      (k, PrimCall(self)),
}

def miranda_show(primval, prim_to_str, bob):
    shown = '' if isinstance(primval, Bob) else prim_to_str(primval)
    slots = list_slots(bob)
    if slots: shown += '{' + ', '.join(sorted(slots)) + '}' 
    return shown


# Primitive object types

number_type = (int, float)
string_type = str               # XXX or unicode, in python2

def prim_add(self, doing, k):
    return call(doing, 'arg1', (add_k, self, k))

def add_k(arg1, self, k):
    if isinstance(arg1, number_type):
        return k, self.ancestor + arg1
    else:
        return call(arg1, 'add_to', (add_to_k, self.arg0, k))

def add_to_k(arg1_add_to, arg0, k):
    return call(Bob(arg1_add_to, {'arg1': lambda _, __, mk: (mk, arg0)}),
                '()', k)

class PrimOp(Bob):
    def __init__(self, ancestor, arg0):
        self.ancestor = ancestor
        self.arg0 = arg0

class PrimAdd(PrimOp):
    name, methods = '+', {
        '()': prim_add
    }

# The other arith ops should also double-dispatch, but for now here
# they are unconverted, since mainly I wanted to make sure it'd work,
# and I don't know what Kragen wants in detail.

class BarePrimOp(Bob):
    methods = {
        '()': lambda self, doing, k: call(doing, 'arg1', (self.arg1_k, self, k))
    }
    def __init__(self, ancestor, arg0):
        self.primval = ancestor

def sub_k(arg1, self, k): return k, self.primval - arg1
def mul_k(arg1, self, k): return k, self.primval * arg1
def div_k(arg1, self, k): return k, self.primval / arg1
def pow_k(arg1, self, k): return k, self.primval ** arg1
def eq_k(arg1, self, k):  return k, self.primval == arg1
def lt_k(arg1, self, k):  return k, self.primval < arg1

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

def string_cat_k(arg1, self, k):
    assert isinstance(arg1, string_type), arg1
    return k, self.primval + arg1
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
    'if': lambda _, me, k: (k, (pick_so if me else pick_else)),
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


# Evaluator plus trivial compiler.
# Glossary:
#   senv: static environment: mapping each variable name to an index
#     into the interpreter's runtime env. (But the compiler uses the
#     Javascript environment instead.)
#   js_foo: represents a javascript expression.
#     Either a string of literal code, or a 3-tuple for an expression to
#     construct a 3-tuple.

def js_compile(expr, js_k='null'):
    return js_render(expr.compile(js_k))

def js_render(py):
    if isinstance(py, str):
        return py
    js_fn, js_free_var, js_k = py
    return '[%s, %s, %s]' % tuple(map(js_render, (js_fn, js_free_var, js_k)))

def js_apply_cont(js_k, py):
    if isinstance(js_k, str):
        return '[%s, %s]' % (js_k, js_render(py))    
    js_fn, js_free_var, js_k = js_k
    if js_fn == 'extendK':
        return js_apply_cont(js_k, 'makeBob(%s, %s)' % (js_render(py),
                                                        js_render(js_free_var)))
    else:
        return '%s(%s, %s, %s)' % (js_render(js_fn),
                                   js_render(py),
                                   js_render(js_free_var),
                                   js_render(js_k))

def js_push_cont(js_fn, js_free_var, js_k):
    return (js_fn, js_free_var, js_k)

def js_name(name):
    # Avoid Bicicleta names shadowing Javascript ones when compiled to
    # Python.
    return '__' if name is None else name + '_b'

js_repr = repr                  # XXX unicode different

def js_slot(slot): return '$' + slot

class VarRef(object):
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return self.name
    def compile(self, js_k):
        return js_apply_cont(js_k, js_name(self.name))
    def analyze(self, senv):
        self.index = senv[self.name]
    def eval(self, env, k):
        return k, env[self.index]

class Literal(object):
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return repr(self.value)
    def compile(self, js_k):
        py = 'rootBob' if self.value is root_bob else js_repr(self.value)
        return js_apply_cont(js_k, py)
    def analyze(self, senv):
        pass
    def eval(self, env, k):
        return k, self.value

class Call(object):
    def __init__(self, receiver, slot):
        self.receiver = receiver
        self.slot = slot
    def __repr__(self):
        return '%s.%s' % (self.receiver, self.slot)
    def compile(self, js_k):
        return self.receiver.compile(js_push_cont('call', js_repr(js_slot(self.slot)), js_k))
    def analyze(self, senv):
        self.receiver.analyze(senv)
    def eval(self, env, k):
        return self.receiver.eval(env, (call, self.slot, k))

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
    def compile(self, js_k):
        me = js_name(self.name)
        methods_expr = (
            '{%s}' % (', '.join('%s: function(_, %s, k) { return %s; }'
                                % (js_repr(js_slot(slot)),
                                   me,
                                   js_render(expr.compile('k')))
                                for slot, expr in self.bindings)))
        assert isinstance(methods_expr, str)
        return self.base.compile(js_push_cont('extendK', methods_expr, js_k))
    def analyze(self, senv):
        self.base.analyze(senv)
        if self.name is not None:
            senv = dict(senv)
            senv[self.name] = len(senv)
        for _, expr in self.bindings:
            expr.analyze(senv)

    def eval(self, env, k):
        methods = {slot: make_selfish_method(expr, env)
                   for slot, expr in self.bindings}
        return self.base.eval(env, (extend_k, methods, k))

class SelflessExtend(Extend):
    def eval(self, env, k):
        methods = {slot: make_selfless_method(expr, env)
                   for slot, expr in self.bindings}
        return self.base.eval(env, (extend_k, methods, k))

def extend_k(bob, methods, k):
    return k, Bob(bob, methods)

def make_selfish_method(expr, env):
    return lambda _, bob, k: expr.eval(env + (bob,), k)

def make_selfless_method(expr, env):
    return lambda _, bob, k: expr.eval(env, k)


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
            | empty extend                  attach

affixes     = affix affixes | 
affix       = _ [.] name                    defer_dot
            | extend
            | _ \( bindings _ \)            defer_funcall
            | _ \[ bindings _ \]            defer_squarecall

extend      = _ { name _ : bindings _ }     defer_extend
            | _ {          bindings _ }     defer_selfless_extend
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

empty_literal = Literal(root_bob)

def empty(): return empty_literal
def positional(): return None

def name_positions(*bindings):
    return tuple((('arg%d' % i if slot is None else slot), expr)
                 for i, (slot, expr) in enumerate(bindings, 1))

def attach_all(expr, *affixes):      return reduce(attach, affixes, expr)
def attach(expr, affix):             return affix[0](expr, *affix[1:])

def defer_dot(name):                 return Call, name
def defer_extend(name, bindings):    return Extend, name, bindings
def defer_selfless_extend(bindings): return SelflessExtend, None, bindings
def defer_funcall(bindings):         return mk_funcall, '()', bindings
def defer_squarecall(bindings):      return mk_funcall, '[]', bindings
def defer_infix(operator, expr):     return mk_infix, operator, expr

def mk_funcall(expr, slot, bindings):
    "  foo(x=y) ==> foo{x=y}.'()'  "
    return Call(SelflessExtend(expr, None, bindings), slot)

def mk_infix(left, operator, right):
    "   x + y ==> x.'+'(_=y)  "
    return mk_funcall(Call(left, operator), '()', (('arg1', right),))

parse = OneResult(Parser(program_grammar, int=int, float=float, **globals()))


# The global environment, standard library, and entry point.

# We incorporate these interpreted methods by mutating the primitive
# method tables, because:
#   * We want to be able to test the system with just the core
#     primitives and no library.
#   * We don't want to slow down the primitive types by extending each
#     'real' primitive with a Bob adding the library methods.

# TODO: add some interpreted miranda methods too.

sys_bob = Bob(None, {
    'true':  lambda _, me, k: (k, True),
    'false': lambda _, me, k: (k, False),
})
global_env = (sys_bob,)
sys_b = sys_bob # For reference by compiled programs.

def run(program, compile=False, dump=False, trace=False):
    if isinstance(program, string_type):
        program = parse(program)
    program.analyze({'sys': 0})
    if compile:
        js = js_compile(program)
        print(js)
        return #state = eval(py)
    else:
        state = program.eval(global_env, None)
    return trampoline(state, trace)

def extend_in_place(methods, overlay):
    # TODO: deep copy? Shallow is all we need for now.
    for slot in overlay.methods:
        assert slot not in methods, slot
    methods.update(overlay.methods)

def load(filename, compile=False):
    return run(open(filename).read(), compile=compile)

#if False:
if True:
    compiling = False
    extend_in_place(bool_methods,    load('sys_bool.bicicleta',   compile=compiling))
    extend_in_place(number_methods,  load('sys_number.bicicleta', compile=compiling))
    extend_in_place(string_methods,  load('sys_string.bicicleta', compile=compiling))
    extend_in_place(sys_bob.methods, load('sys.bicicleta',        compile=compiling))


# Crude tests and benchmarks

## parse('{x: {y: {z: x ++ y{a="b"} <*> z.foo }}}')
#. {x: arg1={y: arg1={z: arg1=x.++{arg1=y{a='b'}}.().<*>{arg1=z.foo}.()}}}

## parse('5')
#. 5

## js_compile(parse('5'), 'k')
#. '[k, 5]'
## js_compile(parse('5+6'), 'null')
#. "call(5, '$+', [extendK, {'$arg1': function(_, __, k) { return [k, 6]; }}, [call, '$()', null]])"

def dump_compile(program):
    return run(program, compile=True, dump=True)

## dump_compile('5+6')
#. call(5, '$+', [extendK, {'$arg1': function(_, __, k) { return [k, 6]; }}, [call, '$()', null]])

## run('5')
#. 5

## wtf = parse("{x=42, y=55}.x")
## wtf
#. {x=42, y=55}.x
## run(wtf)
#. 42

## run("{y=42, x=55, z=137}.x")
#. 55

## parse("137")
#. 137
## parse('137[yo="dude"]')
#. 137{yo='dude'}.[]

## adding = parse("137.'+' {arg1=1}.'()'")
## adding
#. 137.+{arg1=1}.()
## run(adding)
#. 138

## run('3+2')
#. 5
## run('3*2')
#. 6

## run("137.5 - 2 - 1")
#. 134.5

## run(""" 5{is_string=42}.'reflective slot value'("is_string") """)
#. 42
## run(' 5.is_number ')
#. True
## run(' "".is_number ')
#. False
## run(' {}.is_number ')
#. False
## run(' 5{}.is_number ')
#. True

## run("(136 < 137).if(so=1, else=2)")
#. 1
## run("(137 < 137).if(so=1, else=2)")
#. 2
## run("137.'<' {arg1=137}.'()'.if(so=1, else=2)")
#. 2

## cmping = parse("(137 == 1).if(so=42, else=168)")
## repr(cmping) == repr(parse("137.'=='{arg1=1}.'()'.if{so=42, else=168}.'()'"))
#. True
## run(cmping)
#. 168

## run('"howdy"')
#. 'howdy'
## run('("hello" == "aloha").if(so=42, else=168)')
#. 168
## run('("hello" == "hello").if(so=42, else=168)')
#. 42

test_extend = parse("""
    {main:
     three = {me: x = 3, xx = me.x + me.x},
     four = main.three{x=4},
     result = main.three.xx + main.four.xx
    }.result
""")
## run(test_extend)
#. 14
## dump_compile(test_extend)
#. call(makeBob(rootBob, {'$three': function(_, main_b, k) { return [k, makeBob(rootBob, {'$x': function(_, me_b, k) { return [k, 3]; }, '$xx': function(_, me_b, k) { return call(me_b, '$x', [call, '$+', [extendK, {'$arg1': function(_, __, k) { return call(me_b, '$x', k); }}, [call, '$()', k]]]); }})]; }, '$four': function(_, main_b, k) { return call(main_b, '$three', [extendK, {'$x': function(_, __, k) { return [k, 4]; }}, k]); }, '$result': function(_, main_b, k) { return call(main_b, '$three', [call, '$xx', [call, '$+', [extendK, {'$arg1': function(_, __, k) { return call(main_b, '$four', [call, '$xx', k]); }}, [call, '$()', k]]]]); }}), '$result', null)


## run('"hey " ++ 42.str ++ " and " ++ (1136+1).str.rest')
#. 'hey 42 and 137'
# run('"hey {x} and {why}" % {x=84/2, why=136+1}')
## run("5**3")
#. 125

## run("5{}*6")
#. 30

## run("5.is_string")
#. False
## run("5.is_number")
#. True

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
#. 24
## dump_compile(fac)
#. call(makeBob(rootBob, {'$fac': function(_, env_b, k) { return [k, makeBob(rootBob, {'$()': function(_, fac_b, k) { return call(fac_b, '$n', [call, '$==', [extendK, {'$arg1': function(_, __, k) { return [k, 0]; }}, [call, '$()', [call, '$if', [extendK, {'$so': function(_, __, k) { return [k, 1]; }, '$else': function(_, __, k) { return call(fac_b, '$n', [call, '$*', [extendK, {'$arg1': function(_, __, k) { return call(env_b, '$fac', [extendK, {'$n': function(_, __, k) { return call(fac_b, '$n', [call, '$-', [extendK, {'$arg1': function(_, __, k) { return [k, 1]; }}, [call, '$()', k]]]); }}, [call, '$()', k]]); }}, [call, '$()', k]]]); }}, [call, '$()', k]]]]]]); }})]; }}), '$fac', [extendK, {'$n': function(_, __, k) { return [k, 4]; }}, [call, '$()', null]])

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
#. 8


# The following tests depend on the standard library being loaded.

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
