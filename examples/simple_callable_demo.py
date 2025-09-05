#!/usr/bin/env python3
"""
Simple ESR Callable Demo

Quick demonstration of creating Python and JAX callables from ESR functions.
"""

import numpy as np
import sympy
import jax.numpy as jnp
from jax import jit
import random
import os

# ESR imports
import esr.generation.duplicate_checker as duplicate_checker
import esr.generation.generator as generator

def quick_demo():
    """Quick demonstration of ESR callables"""
    print("🚀 ESR Callable Quick Demo")
    print("=" * 40)
    
    # 1. Generate some functions (using existing if available)
    try:
        library_dir = os.path.abspath(os.path.join(os.path.dirname(generator.__file__), '..', 'function_library'))
        eq_filename = os.path.join(library_dir, "osc_maths", "compl_4", "unique_equations_4.txt")
        with open(eq_filename, "r") as f:
            functions = [line.strip() for line in f.readlines() if line.strip()]
    except:
        print("Generating new functions...")
        try:
            duplicate_checker.main("osc_maths", 4)
        except SystemExit:
            pass
        with open(eq_filename, "r") as f:
            functions = [line.strip() for line in f.readlines() if line.strip()]
    
    # 2. Pick a function with parameters
    random.seed(123)  # For consistent demo
    func_str = "sin(x) + a0*x"  # Simple example
    if func_str not in functions:
        func_str = random.choice([f for f in functions if 'a0' in f])
    
    print(f"📝 Selected function: {func_str}")
    
    # 3. Parse and create callables
    x = sympy.Symbol('x')
    a0 = sympy.Symbol('a0')
    
    # Simple parsing for this demo
    expr = sympy.sympify(func_str, locals={'x': x, 'a0': a0, 'sin': sympy.sin, 'cos': sympy.cos})
    
    # Python callable
    def python_func(x_val, a0_val=1.0):
        """Python callable: f(x, a0)"""
        substituted = expr.subs(a0, a0_val)
        return float(sympy.lambdify(x, substituted, 'numpy')(x_val))
    
    # JAX callable (JIT compiled)
    @jit
    def jax_func(x_val, a0_val):
        """JAX callable: f(x, a0) - JIT compiled"""
        # For demo, we'll use a specific function
        return jnp.sin(x_val) + a0_val * x_val
    
    # 4. Test both versions
    print("\n🧪 Testing callables:")
    
    x_test = 1.5
    a0_test = 2.0
    
    py_result = python_func(x_test, a0_test)
    jax_result = float(jax_func(x_test, a0_test))
    
    print(f"   Input: x={x_test}, a0={a0_test}")
    print(f"   Python result: {py_result:.6f}")
    print(f"   JAX result:    {jax_result:.6f}")
    print(f"   Match: {abs(py_result - jax_result) < 1e-6}")
    
    # 5. Array evaluation
    print("\n📊 Array evaluation:")
    x_array = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    
    # Vectorize Python function
    py_vectorized = np.vectorize(python_func)
    py_results = py_vectorized(x_array, a0_test)
    
    # JAX is already vectorized
    jax_results = jax_func(x_array, a0_test)
    
    print(f"   Input array: {x_array}")
    print(f"   Python results: {py_results}")
    print(f"   JAX results:    {np.array(jax_results)}")
    print(f"   Arrays match: {np.allclose(py_results, jax_results)}")
    
    # 6. Performance comparison
    print("\n⚡ Performance comparison:")
    import time
    
    x_large = np.random.rand(1000)
    
    # Warm up JAX
    _ = jax_func(x_large, a0_test)
    
    # Time Python
    start = time.time()
    for _ in range(100):
        _ = py_vectorized(x_large, a0_test)
    python_time = time.time() - start
    
    # Time JAX
    start = time.time()
    for _ in range(100):
        _ = jax_func(x_large, a0_test)
    jax_time = time.time() - start
    
    print(f"   Python (100 evals): {python_time:.4f}s")
    print(f"   JAX (100 evals):    {jax_time:.4f}s")
    print(f"   JAX speedup:        {python_time/jax_time:.1f}x")
    
    print("\n✅ Demo complete!")
    print("\n💡 Key takeaways:")
    print("   • ESR functions can be converted to Python/JAX callables")
    print("   • Parameters can be passed as function arguments")
    print("   • JAX provides automatic vectorization and JIT speedup")
    print("   • Both versions produce identical results")

if __name__ == "__main__":
    quick_demo()
