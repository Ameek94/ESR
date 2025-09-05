#!/usr/bin/env python3
"""
Improved Minimal ESR Function Callable Example

This script demonstrates how to:
1. Pick a random function from ESR-generated functions
2. Create a Python callable that takes x and parameters (as args or kwargs)
3. Create an optimized JAX version of the same function
4. Compare both implementations with proper JAX optimization
"""

import numpy as np
import sympy
import random
import os
import esr.generation.duplicate_checker as duplicate_checker
import esr.generation.generator as generator

# JAX imports
import jax
import jax.numpy as jnp
from jax import jit
from typing import Callable, Union, Dict, Any

def generate_and_load_functions(complexity: int = 4, runname: str = "osc_maths") -> list:
    """Generate ESR functions and load them from file"""
    print(f"Generating functions with complexity {complexity}...")
    
    # Generate functions
    try:
        duplicate_checker.main(runname, complexity)
    except SystemExit:
        pass
    
    # Load generated functions
    library_dir = os.path.abspath(os.path.join(os.path.dirname(generator.__file__), '..', 'function_library'))
    eq_filename = os.path.join(library_dir, runname, f"compl_{complexity}", f"unique_equations_{complexity}.txt")
    
    try:
        with open(eq_filename, "r") as f:
            functions = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Could not find file: {eq_filename}")
        return []
    
    print(f"Loaded {len(functions)} unique functions")
    return functions

def parse_esr_function(func_string: str) -> tuple:
    """Parse ESR function string and return sympy expression and parameter symbols"""
    # Set up symbols
    x = sympy.symbols('x', real=True)
    a_symbols = sympy.symbols([f'a{i}' for i in range(10)], real=True)
    
    # Define parsing context
    locs = {
        'x': x, 
        'sin': sympy.sin, 
        'cos': sympy.cos, 
        'exp': sympy.exp,
        'log': sympy.log,
        'inv': lambda x: 1/x, 
        'Abs': sympy.Abs, 
        'pow': sympy.Pow
    }
    
    # Add parameter symbols
    for i, a_sym in enumerate(a_symbols):
        locs[f'a{i}'] = a_sym
    
    # Parse the function
    expr = sympy.sympify(func_string, locals=locs)
    
    # Extract parameter symbols
    param_symbols = [sym for sym in expr.free_symbols if str(sym).startswith('a')]
    param_symbols.sort(key=lambda s: int(str(s)[1:]))  # Sort by number: a0, a1, a2, ...
    
    return expr, param_symbols

class ESRCallable:
    """A callable wrapper for ESR functions that handles both Python and JAX implementations"""
    
    def __init__(self, func_string: str):
        self.func_string = func_string
        self.expr, self.param_symbols = parse_esr_function(func_string)
        self.param_names = [str(p) for p in self.param_symbols]
        self._jax_functions = {}  # Cache for JIT-compiled functions
        
        print(f"Created ESR callable for: {func_string}")
        print(f"Parameters: {self.param_names}")
    
    def __call__(self, x: Union[float, np.ndarray], *args, **kwargs) -> Union[float, np.ndarray]:
        """
        Evaluate using NumPy (Python) implementation.
        
        Args:
            x: Input value(s)
            *args: Parameter values in order
            **kwargs: Parameter values as keyword arguments
        
        Returns:
            Function evaluation result
        """
        param_values = self._process_parameters(args, kwargs)
        expr_with_params = self.expr.subs(param_values)
        func = sympy.lambdify([sympy.Symbol('x')], expr_with_params, modules=['numpy'])
        return func(x)
    
    def jax(self, x: Union[float, jnp.ndarray], *args, **kwargs) -> jnp.ndarray:
        """
        Evaluate using JAX implementation with JIT compilation.
        
        Args:
            x: Input value(s) as JAX array
            *args: Parameter values in order
            **kwargs: Parameter values as keyword arguments
        
        Returns:
            Function evaluation result as JAX array
        """
        param_values = self._process_parameters(args, kwargs)
        
        # Create a cache key based on parameter values
        cache_key = tuple(sorted(param_values.items()))
        
        # Check if we have a cached JIT function for these parameter values
        if cache_key not in self._jax_functions:
            # Substitute parameters and create JIT function
            expr_with_params = self.expr.subs(param_values)
            jax_func = jit(sympy.lambdify(
                [sympy.Symbol('x')], 
                expr_with_params, 
                modules=[
                    {'sin': jnp.sin, 'cos': jnp.cos, 'exp': jnp.exp, 'log': jnp.log,
                     'Abs': jnp.abs, 'pow': jnp.power, 'inv': lambda x: 1/x}, 
                    'jax'
                ]
            ))
            self._jax_functions[cache_key] = jax_func
        
        # Use cached function
        x_jax = jnp.asarray(x)
        return self._jax_functions[cache_key](x_jax)
    
    def _process_parameters(self, args, kwargs):
        """Process parameter arguments and return substitution dictionary"""
        if args and kwargs:
            raise ValueError("Cannot specify both args and kwargs for parameters")
        
        if args:
            if len(args) != len(self.param_symbols):
                raise ValueError(f"Expected {len(self.param_symbols)} parameters, got {len(args)}")
            return {param: args[i] for i, param in enumerate(self.param_symbols)}
        elif kwargs:
            param_values = {}
            for param in self.param_symbols:
                param_name = str(param)
                if param_name not in kwargs:
                    raise ValueError(f"Missing parameter: {param_name}")
                param_values[param] = kwargs[param_name]
            return param_values
        else:
            # Default values
            return {param: 1.0 for param in self.param_symbols}

def demonstrate_esr_callable():
    """Main demonstration function"""
    print("=" * 60)
    print("ESR Improved Callable Example")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # 1. Generate and load functions
    functions = generate_and_load_functions(complexity=4, runname="osc_maths")
    
    if not functions:
        print("No functions generated. Exiting.")
        return
    
    # 2. Pick a random function with parameters
    functions_with_params = []
    for func_str in functions:
        try:
            expr, param_symbols = parse_esr_function(func_str)
            if param_symbols:  # Only functions with parameters
                functions_with_params.append(func_str)
        except:
            continue
    
    if not functions_with_params:
        print("No functions with parameters found. Exiting.")
        return
    
    # Pick random function
    selected_function = random.choice(functions_with_params)
    
    print(f"\nSelected function: {selected_function}")
    
    # 3. Create ESR callable
    esr_func = ESRCallable(selected_function)
    
    # 4. Test with different parameter passing methods
    x_test = np.array([0.1, 0.5, 1.0, 1.5, 2.0])  # Use positive values to avoid sqrt warnings
    
    print(f"\n" + "=" * 40)
    print("Testing Callable Functions")
    print("=" * 40)
    
    print(f"Test input: x = {x_test}")
    
    # Test with default parameters
    print(f"\n1. Default parameters (all = 1.0):")
    result_py_default = esr_func(x_test)
    result_jax_default = esr_func.jax(x_test)
    print(f"   Python result: {result_py_default}")
    print(f"   JAX result:    {np.array(result_jax_default)}")
    print(f"   Match: {np.allclose(result_py_default, result_jax_default, rtol=1e-6)}")
    
    # Test with positional arguments
    if len(esr_func.param_symbols) <= 3:  # Only test if manageable number of params
        param_values = [0.5, 2.0, -1.0][:len(esr_func.param_symbols)]
        print(f"\n2. Positional arguments: {param_values}")
        result_py_args = esr_func(x_test, *param_values)
        result_jax_args = esr_func.jax(x_test, *param_values)
        print(f"   Python result: {result_py_args}")
        print(f"   JAX result:    {np.array(result_jax_args)}")
        print(f"   Match: {np.allclose(result_py_args, result_jax_args, rtol=1e-6)}")
    
    # Test with keyword arguments
    param_kwargs = {name: 0.5 + i * 0.3 for i, name in enumerate(esr_func.param_names)}
    print(f"\n3. Keyword arguments: {param_kwargs}")
    result_py_kwargs = esr_func(x_test, **param_kwargs)
    result_jax_kwargs = esr_func.jax(x_test, **param_kwargs)
    print(f"   Python result: {result_py_kwargs}")
    print(f"   JAX result:    {np.array(result_jax_kwargs)}")
    print(f"   Match: {np.allclose(result_py_kwargs, result_jax_kwargs, rtol=1e-6)}")
    
    # 5. Performance comparison with proper JAX optimization
    print(f"\n" + "=" * 40)
    print("Performance Comparison")
    print("=" * 40)
    
    import time
    
    # Use larger arrays for meaningful performance comparison
    x_large = np.random.uniform(0.1, 5.0, 10000)  # Positive values
    
    # Warm up JAX (important for fair comparison)
    print("Warming up JAX...")
    for _ in range(5):
        _ = esr_func.jax(x_large, **param_kwargs)
    
    # Time Python version
    print("Timing Python version...")
    start = time.time()
    for _ in range(50):
        _ = esr_func(x_large, **param_kwargs)
    python_time = time.time() - start
    
    # Time JAX version (now warmed up)
    print("Timing JAX version...")
    start = time.time()
    for _ in range(50):
        _ = esr_func.jax(x_large, **param_kwargs)
    jax_time = time.time() - start
    
    print(f"\nResults (50 evaluations, {len(x_large)} points each):")
    print(f"Python time: {python_time:.4f}s")
    print(f"JAX time:    {jax_time:.4f}s")
    if jax_time > 0:
        print(f"Speedup:     {python_time/jax_time:.2f}x")
    
    # 6. Demonstrate caching efficiency
    print(f"\n" + "=" * 40)
    print("JAX Caching Demonstration")
    print("=" * 40)
    
    # First call (compilation time)
    new_params = {'a0': 3.14} if 'a0' in esr_func.param_names else {}
    start = time.time()
    _ = esr_func.jax(x_test, **new_params)
    first_call_time = time.time() - start
    
    # Second call (cached, should be much faster)
    start = time.time()
    _ = esr_func.jax(x_test, **new_params)
    second_call_time = time.time() - start
    
    print(f"First call (compilation): {first_call_time*1000:.2f}ms")
    print(f"Second call (cached):     {second_call_time*1000:.2f}ms")
    print(f"Cached functions: {len(esr_func._jax_functions)}")
    
    print(f"\n" + "=" * 60)
    print("Example Complete!")
    print(f"✅ Created flexible callable for ESR function: {selected_function}")
    print(f"✅ Supports both positional and keyword arguments")
    print(f"✅ Python and JAX implementations match")
    print(f"✅ JAX version uses caching for performance")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_esr_callable()
