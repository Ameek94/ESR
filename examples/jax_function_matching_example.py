import numpy as np
import sympy
import matplotlib.pyplot as plt
import esr.generation.duplicate_checker as duplicate_checker
import os
import random
from scipy.optimize import minimize

# JAX imports
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from typing import Callable

# Set random seed for reproducible comparison
np.random.seed(42)
random.seed(42)
    # JAX imports
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from typing import Callable

# Set random seed for reproducible comparison
np.random.seed(42)
random.seed(42)

def sympy_to_jax(expr: sympy.Expr, variables: list) -> Callable:
    """
    Convert a SymPy expression to a JAX function.
    
    Args:
        expr: SymPy expression
        variables: List of SymPy symbols that are the function variables
        
    Returns:
        JAX function that can be jitted and differentiated
    """
    # Check for problematic expressions first
    expr_str = str(expr)
    if 'ComplexInfinity' in expr_str or 'zoo' in expr_str or 'nan' in expr_str:
        raise ValueError(f"Expression contains problematic terms: {expr_str}")
    
    try:
        # Convert SymPy expression to JAX using sympy.lambdify with 'jax' modules
        jax_func = sympy.lambdify(variables, expr, modules=[
            {'sin': jnp.sin, 'cos': jnp.cos, 'exp': jnp.exp, 'log': jnp.log,
             'Abs': jnp.abs, 'pow': jnp.power, 'inv': lambda x: 1/x}, 
            'jax'
        ])
        
        # Test the function with a simple input to ensure it works
        test_input = jnp.array(1.0)
        test_result = jax_func(test_input)
        
        # Check for invalid results
        if jnp.isnan(test_result) or jnp.isinf(test_result):
            raise ValueError("Function produces invalid results")
            
        return jax_func
        
    except Exception as e:
        # If JAX conversion fails, raise an error to skip this function
        raise ValueError(f"Error converting to JAX: {e}")

def create_jax_objective_function(expr_template, param_symbols, x_vals, y_target):
    """Create a JAX-based objective function for parameter optimization with scipy"""
    
    def objective(params):
        try:
            # Create substitutions dictionary
            substitutions = {param: params[i] for i, param in enumerate(param_symbols)}
            
            # Substitute parameters into expression  
            expr_with_params = expr_template.subs(substitutions)
            
            # Check for problematic expressions before conversion
            expr_str = str(expr_with_params)
            if 'ComplexInfinity' in expr_str or 'zoo' in expr_str or 'nan' in expr_str:
                return 1e10  # Large penalty for problematic expressions
            
            # Convert to JAX function
            jax_func = sympy_to_jax(expr_with_params, [sympy.Symbol('x')])
            
            # Evaluate function
            y_pred = jax_func(x_vals)
            
            # Check for invalid values
            if jnp.any(jnp.isinf(y_pred)) or jnp.any(jnp.isnan(y_pred)):
                return 1e10  # Large penalty for invalid values
            
            # Return mean squared error as float for scipy
            return float(jnp.mean((y_pred - y_target) ** 2))
        except:
            return 1e10  # Large penalty for any errors
    
    return objective

# 1. Set up parameters for function generation
complexity = 6  # Match function_matching_example for fair comparison
runname = "osc_maths"

print(f"Generating functions with complexity {complexity} for run '{runname}'...")

# 2. Generate and process equations using the main entry point from the library
try:
    duplicate_checker.main(runname, complexity)
except SystemExit:
    pass

print("\nFunction generation and processing complete.")

# 3. Read the generated functions from the file
import esr.generation.generator as generator
library_dir = os.path.abspath(os.path.join(os.path.dirname(generator.__file__), '..', 'function_library'))
eq_filename = os.path.join(library_dir, runname, f"compl_{complexity}", f"unique_equations_{complexity}.txt")
try:
    with open(eq_filename, "r") as f:
        all_functions = [line.strip() for line in f.readlines() if line.strip()]
except FileNotFoundError:
    print(f"Could not find file with generated equations: {eq_filename}")
    exit()

if not all_functions:
    print("No functions were generated. Exiting.")
    exit()

print(f"Found {len(all_functions)} unique functions.")

# 4. Define the true function we want to match
def true_function(x):
    return jnp.sin(x) * x + 0.5 * x

# Create evaluation points in the interval [0, 2π]
x_eval = jnp.linspace(0, 2*jnp.pi, 100)
y_true = true_function(x_eval) + np.random.normal(0, 0.01, size=x_eval.shape)  # Match function_matching_example noise level

print(f"\nTarget function: y = x * sin(x) + 0.5x")
print(f"Evaluating functions over interval [0, 2π] with {len(x_eval)} points")

# 5. Evaluate functions with JAX
best_function = None
best_error = float('inf')
best_jax_func = None
best_expr = None
best_params = None
valid_functions = 0
errors = []
function_results = []

print(f"\nEvaluating {len(all_functions)} functions with JAX and scipy optimization...")

# Use all functions for fair comparison with function_matching_example
for i, func_string in enumerate(all_functions):
    if (i + 1) % 25 == 0:
        print(f"Processed {i + 1}/{len(all_functions)} functions...")
    
    try:
        # Convert string to sympy expression
        x = sympy.symbols('x', real=True)
        a_symbols = sympy.symbols([f'a{i}' for i in range(10)], real=True)
        
        locs = {'x': x, 'sin': sympy.sin, 'cos': sympy.cos, 'inv': lambda x: 1/x, 
                'Abs': sympy.Abs, 'pow': sympy.Pow, 'exp': sympy.exp, 'log': sympy.log}
        
        for j, a_sym in enumerate(a_symbols):
            locs[f'a{j}'] = a_sym

        expr_template = sympy.sympify(func_string, locals=locs)
        param_symbols = [sym for sym in expr_template.free_symbols if str(sym).startswith('a')]
        
        if param_symbols:
            # Optimize parameters using scipy with JAX-based objective and multiple restarts
            # Create JAX-based objective function
            objective = create_jax_objective_function(expr_template, param_symbols, x_eval, y_true)
            
            # Optimize parameters with bounds to prevent extreme values
            bounds = [(-10, 10) for _ in param_symbols]  # Same bounds as function_matching_example
            
            # Multiple restart optimization to avoid local minima
            best_result = None
            best_mse = float('inf')
            
            # Try different starting points for robust optimization
            num_params = len(param_symbols)
            starting_points = [
                np.ones(num_params),           # Start at 1.0
                np.zeros(num_params),          # Start at 0.0
                np.full(num_params, 0.5),      # Start at 0.5
                np.full(num_params, -0.5),     # Start at -0.5
                np.random.uniform(-2, 2, num_params),  # Random start 1
                np.random.uniform(-1, 1, num_params),  # Random start 2
            ]
            
            try:
                for start_point in starting_points:
                    try:
                        result = minimize(objective, start_point, bounds=bounds, method='L-BFGS-B', 
                                        options={'maxfun': 200})  # Reduced maxfun per restart
                        
                        if result.success and result.fun < best_mse:
                            best_result = result
                            best_mse = result.fun
                            
                    except:
                        continue
                
                # Also try gradient-free method as backup
                try:
                    result_nm = minimize(objective, np.ones(num_params), method='Nelder-Mead', 
                                       options={'maxfev': 200})
                    
                    if result_nm.success and result_nm.fun < best_mse:
                        best_result = result_nm
                        best_mse = result_nm.fun
                        
                except:
                    pass
                
                if best_result and best_result.success:
                    # Use best optimized parameters
                    optimized_params = best_result.x
                    mse = best_result.fun
                    
                    # Create final expression with optimized parameters
                    substitutions = {param: optimized_params[j] for j, param in enumerate(param_symbols)}
                    final_expr = expr_template.subs(substitutions)
                    final_jax_func = sympy_to_jax(final_expr, [x])
                    
                else:
                    # All optimization attempts failed, skip this function
                    continue
                    
            except Exception as e:
                # Optimization failed, skip this function
                continue
                
        else:
            # No parameters to optimize
            final_expr = expr_template
            final_jax_func = sympy_to_jax(final_expr, [x])
            
            try:
                y_pred = final_jax_func(x_eval)
                
                # Check for invalid values
                if jnp.any(jnp.isinf(y_pred)) or jnp.any(jnp.isnan(y_pred)):
                    continue
                    
                mse = float(jnp.mean((y_pred - y_true) ** 2))
                
            except Exception as e:
                continue
        
        # Record results
        errors.append(mse)
        function_results.append((mse, func_string, final_expr))
        
        if mse < best_error:
            best_error = mse
            best_function = func_string
            best_jax_func = final_jax_func
            best_expr = final_expr
            if param_symbols:
                best_params = optimized_params
            else:
                best_params = None
            
        valid_functions += 1
            
    except Exception as e:
        continue

print(f"\nEvaluation complete!")
print(f"Successfully evaluated {valid_functions} out of {len(all_functions)} functions")

if best_function is None:
    print("No valid function found that could be evaluated.")
    exit()

print(f"\nBest matching function:")
print(f"Function string: {best_function}")
print(f"Optimized expression: {best_expr}")
print(f"Mean Squared Error: {best_error:.6f}")

# 6. Demonstrate JAX capabilities with the best function

# JIT compilation
@jit
def jitted_best_func(x):
    return best_jax_func(x)

# Gradient computation - use vmap to vectorize over inputs
grad_func_scalar = jit(grad(best_jax_func))
grad_func = jit(vmap(grad_func_scalar))

# Test evaluation
x_test = jnp.array(1.0)
print(f"\nJAX capabilities demonstration:")
print(f"Function value at x=1: {jitted_best_func(x_test):.6f}")
print(f"Gradient at x=1: {grad_func_scalar(x_test):.6f}")

# Vectorized evaluation with JIT
x_plot = jnp.linspace(0, 2*jnp.pi, 200)
y_true_plot = true_function(x_plot)
y_best_plot = jitted_best_func(x_plot)
gradients = grad_func(x_plot)

# 7. Plot the results
plt.figure(figsize=(15, 10))

# Main function plot
plt.subplot(3, 1, 1)
plt.plot(x_plot, y_true_plot, 'b-', linewidth=2, label='True function: x * sin(x) + 0.5x')
plt.plot(x_plot, y_best_plot, 'r--', linewidth=2, label=f'Best ESR match: {best_function}')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'JAX Function Matching: ESR vs True Function (MSE = {best_error:.6f})')
plt.legend()
plt.grid(True)
plt.xlim(0, 2*jnp.pi)

# Error plot
plt.subplot(3, 1, 2)
error_plot = jnp.abs(y_true_plot - y_best_plot)
plt.plot(x_plot, error_plot, 'g-', linewidth=1.5, label='Absolute Error')
plt.xlabel('x')
plt.ylabel('|True - Predicted|')
plt.title('Absolute Error Between True and Best ESR Function')
plt.legend()
plt.grid(True)
plt.xlim(0, 2*jnp.pi)

# Gradient plot
plt.subplot(3, 1, 3)
plt.plot(x_plot, gradients, 'm-', linewidth=1.5, label='Gradient (computed with JAX)')
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('Gradient of Best ESR Function (computed with JAX autodiff)')
plt.legend()
plt.grid(True)
plt.xlim(0, 2*jnp.pi)

plt.tight_layout()
plot_filename = 'jax_function_matching_example.png'
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.show()

print(f"\nPlot saved as {plot_filename}")

# 8. Performance comparison
print(f"\nPerformance comparison:")

# Time regular NumPy function
import time

# Convert best function to NumPy for comparison
numpy_func = sympy.lambdify([x], best_expr, modules=['numpy'])

# Warm up JIT
_ = jitted_best_func(x_plot)

# Time NumPy version
start_time = time.time()
for _ in range(100):
    _ = numpy_func(x_plot)
numpy_time = time.time() - start_time

# Time JAX JIT version  
start_time = time.time()
for _ in range(100):
    _ = jitted_best_func(x_plot)
jax_time = time.time() - start_time

print(f"NumPy function time (100 evaluations): {numpy_time:.6f} seconds")
print(f"JAX JIT function time (100 evaluations): {jax_time:.6f} seconds")
print(f"Speedup: {numpy_time/jax_time:.2f}x")

# 9. Show statistics
if len(errors) > 0:
    errors = np.array(errors)
    print(f"\nStatistics for all {len(errors)} valid functions:")
    print(f"Best (lowest) MSE: {np.min(errors):.6f}")
    print(f"Worst (highest) MSE: {np.max(errors):.6f}")
    print(f"Mean MSE: {np.mean(errors):.6f}")
    print(f"Median MSE: {np.median(errors):.6f}")

print(f"\nExample complete. This demonstrates how to:")
print(f"1. Convert SymPy expressions from ESR to JAX functions")
print(f"2. Use JIT compilation for faster evaluation")
print(f"3. Compute gradients using JAX autodiff")
print(f"4. Leverage JAX for vectorized operations")
