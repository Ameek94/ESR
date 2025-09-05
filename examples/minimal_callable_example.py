#!/usr/bin/env python3
"""
Minimal ESR Function Callable Example

This script demonstrates how to:
1. Pick a random function from ESR-generated functions
2. Create a Python callable that takes x and parameters (as args or kwargs)
3. Create a JAX version of the same function
4. Compare both implementations
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

def create_python_callable(expr: sympy.Expr, param_symbols: list) -> Callable:
    """Create a Python callable that accepts x and parameters"""
    
    def python_func(x: Union[float, np.ndarray], *args, **kwargs) -> Union[float, np.ndarray]:
        """
        Evaluate the function at x with given parameters.
        
        Args:
            x: Input value(s)
            *args: Parameter values in order (a0, a1, a2, ...)
            **kwargs: Parameter values as keyword arguments (a0=val, a1=val, ...)
        
        Returns:
            Function evaluation result
        """
        # Handle parameter values
        if args and kwargs:
            raise ValueError("Cannot specify both args and kwargs for parameters")
        
        if args:
            if len(args) != len(param_symbols):
                raise ValueError(f"Expected {len(param_symbols)} parameters, got {len(args)}")
            param_values = {param: args[i] for i, param in enumerate(param_symbols)}
        elif kwargs:
            param_values = {}
            for param in param_symbols:
                param_name = str(param)
                if param_name not in kwargs:
                    raise ValueError(f"Missing parameter: {param_name}")
                param_values[param] = kwargs[param_name]
        else:
            # Default values
            param_values = {param: 1.0 for param in param_symbols}
        
        # Substitute parameters and create callable
        expr_with_params = expr.subs(param_values)
        func = sympy.lambdify([sympy.Symbol('x')], expr_with_params, modules=['numpy'])
        
        return func(x)
    
    return python_func

def create_jax_callable(expr: sympy.Expr, param_symbols: list) -> Callable:
    """Create a JAX callable that accepts x and parameters"""
    
    def jax_func(x: Union[float, jnp.ndarray], *args, **kwargs) -> jnp.ndarray:
        """
        JAX version of the function.
        
        Args:
            x: Input value(s) as JAX array
            *args: Parameter values in order (a0, a1, a2, ...)
            **kwargs: Parameter values as keyword arguments (a0=val, a1=val, ...)
        
        Returns:
            Function evaluation result as JAX array
        """
        # Handle parameter values
        if args and kwargs:
            raise ValueError("Cannot specify both args and kwargs for parameters")
        
        if args:
            if len(args) != len(param_symbols):
                raise ValueError(f"Expected {len(param_symbols)} parameters, got {len(args)}")
            param_values = {param: args[i] for i, param in enumerate(param_symbols)}
        elif kwargs:
            param_values = {}
            for param in param_symbols:
                param_name = str(param)
                if param_name not in kwargs:
                    raise ValueError(f"Missing parameter: {param_name}")
                param_values[param] = kwargs[param_name]
        else:
            # Default values
            param_values = {param: 1.0 for param in param_symbols}
        
        # Substitute parameters (this happens outside JIT)
        expr_with_params = expr.subs(param_values)
        
        # Convert to JAX function and JIT compile
        jax_func_inner = jit(sympy.lambdify(
            [sympy.Symbol('x')], 
            expr_with_params, 
            modules=[
                {'sin': jnp.sin, 'cos': jnp.cos, 'exp': jnp.exp, 'log': jnp.log,
                 'Abs': jnp.abs, 'pow': jnp.power, 'inv': lambda x: 1/x}, 
                'jax'
            ]
        ))
        
        # Ensure x is a JAX array
        x_jax = jnp.asarray(x)
        return jax_func_inner(x_jax)
    
    return jax_func

def demonstrate_callables():
    """Main demonstration function"""
    print("=" * 60)
    print("ESR Minimal Callable Example")
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
                functions_with_params.append((func_str, expr, param_symbols))
        except:
            continue
    
    if not functions_with_params:
        print("No functions with parameters found. Exiting.")
        return
    
    # Pick random function
    func_str, expr, param_symbols = random.choice(functions_with_params)
    
    print(f"\nSelected random function:")
    print(f"Function string: {func_str}")
    print(f"Parsed expression: {expr}")
    print(f"Parameters: {[str(p) for p in param_symbols]}")
    
    # 3. Create callables
    python_func = create_python_callable(expr, param_symbols)
    jax_func = create_jax_callable(expr, param_symbols)
    
    # 4. Test with different parameter passing methods
    x_test = np.array([0.5, 1.0, 1.5])
    
    print(f"\n" + "=" * 40)
    print("Testing Callable Functions")
    print("=" * 40)
    
    print(f"Test input: x = {x_test}")
    
    # Test with default parameters
    print(f"\n1. Default parameters (all = 1.0):")
    result_py_default = python_func(x_test)
    result_jax_default = jax_func(x_test)
    print(f"   Python result: {result_py_default}")
    print(f"   JAX result:    {np.array(result_jax_default)}")
    print(f"   Match: {np.allclose(result_py_default, result_jax_default)}")
    
    # Test with positional arguments
    if len(param_symbols) <= 3:  # Only test if manageable number of params
        param_values = [0.5, 2.0, -1.0][:len(param_symbols)]
        print(f"\n2. Positional arguments: {param_values}")
        result_py_args = python_func(x_test, *param_values)
        result_jax_args = jax_func(x_test, *param_values)
        print(f"   Python result: {result_py_args}")
        print(f"   JAX result:    {np.array(result_jax_args)}")
        print(f"   Match: {np.allclose(result_py_args, result_jax_args)}")
    
    # Test with keyword arguments
    param_kwargs = {str(param): 0.5 + i * 0.3 for i, param in enumerate(param_symbols)}
    print(f"\n3. Keyword arguments: {param_kwargs}")
    result_py_kwargs = python_func(x_test, **param_kwargs)
    result_jax_kwargs = jax_func(x_test, **param_kwargs)
    print(f"   Python result: {result_py_kwargs}")
    print(f"   JAX result:    {np.array(result_jax_kwargs)}")
    print(f"   Match: {np.allclose(result_py_kwargs, result_jax_kwargs)}")
    
    # 5. Performance comparison
    print(f"\n" + "=" * 40)
    print("Performance Comparison")
    print("=" * 40)
    
    import time
    
    x_large = np.random.randn(1000)
    
    # Warm up JAX
    _ = jax_func(x_large, **param_kwargs)
    
    # Time Python version
    start = time.time()
    for _ in range(100):
        _ = python_func(x_large, **param_kwargs)
    python_time = time.time() - start
    
    # Time JAX version
    start = time.time()
    for _ in range(100):
        _ = jax_func(x_large, **param_kwargs)
    jax_time = time.time() - start
    
    print(f"Python time (100 evals): {python_time:.4f}s")
    print(f"JAX time (100 evals):    {jax_time:.4f}s")
    print(f"Speedup:                 {python_time/jax_time:.2f}x")
    
    print(f"\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_callables()
