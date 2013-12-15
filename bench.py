"""
Quick and dirty benchmarks of the core interpreter.
"""

#import terp
from core import *

def make_tak():
    program = parse("""
{env:
 tak = {tak: 
          '()' = (tak.y < (tak.x)).if(
              so = env.tak(x=env.tak(x=tak.x-1, y=tak.y, z=tak.z),
                           y=env.tak(x=tak.y-1, y=tak.z, z=tak.x),
                           z=env.tak(x=tak.z-1, y=tak.x, z=tak.y)),
              else = tak.z)
         }
    }.tak(x=18, y=12, z=6)""")
    return program

def make_tarai():
    # TARAI is like TAK, but it's much faster with lazy evaluation.
    # It was Takeuchi's original function.
    program = parse("""
{env:
 tarai = {tarai: 
          '()' = (tarai.y < (tarai.x)).if(
              so = env.tarai(x=env.tarai(x=tarai.x-1, y=tarai.y, z=tarai.z),
                             y=env.tarai(x=tarai.y-1, y=tarai.z, z=tarai.x),
                             z=env.tarai(x=tarai.z-1, y=tarai.x, z=tarai.y)),
              else = tarai.y)
         }
    }.tarai(x=18, y=12, z=6)""")
    return program

itersum3 = parse("""
{env:
 outer = {outer: 
   i=0, sum=0,
   '()' = (outer.i == 0).if(
       so = outer.sum,
       else = env.outer(i=outer.i-1, sum=outer.sum + env.mid(i=outer.i)))},
 mid = {mid: 
   i=0, sum=0,
   '()' = (mid.i == 0).if(
       so = mid.sum,
       else = env.mid(i=mid.i-1, sum=mid.sum + env.inner(i=mid.i)))},
 inner = {inner:
   i=0, sum=0,
   '()' = (inner.i == 0).if(
       so = inner.sum,
       else = env.inner(i=inner.i-1, sum=inner.sum + inner.i))},
 main = env.outer(i = 40)
}.main
""")

def timed(f):
    import time
    start = time.clock()
    result = f()
    return time.clock() - start, result

def bench2():
    tarai = make_tarai()
    print(timed(lambda: run(tarai)))

def bench3():
    tak = make_tak()
    print(timed(lambda: run(tak)))

if __name__ == '__main__':
    bench2()
    print(timed(lambda: run(itersum3)))
    fib = make_fib(20)
    print(timed(lambda: run(fib)))
    bench3()
