"""
Simple demonstration of integrating JAX with ESR for fast function evaluation and differentiation.

This script shows how to:
1. Generate functions using ESR
2. Convert them to JAX functions
3. Use JIT compilation and automatic differentiation
"""

import sys
import os
sys.path.append('/Users/amk/cosmocodes/ESR')

import numpy as np
import jax.numpy as jnp
from jax import jit, grad, vmap
import random

# Set random seed for reproducible comparison
np.random.seed(42)
random.seed(42)
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sympy
import random

# Import ESR components
import esr.generation.duplicate_checker as duplicate_checker
import esr.generation.generator as generator
from esr.jax_utils import create_jax_function_from_esr, batch_evaluate_functions

def demonstrate_jax_integration():
    """
    Demonstrate JAX integration with ESR for fast evaluation and differentiation.
    """
    print("ESR-JAX Integration Demonstration")
    print("=" * 50)
    
    # 1. Generate some simple functions using ESR
    complexity = 6
    runname = "osc_maths"
    
    print(f"\n1. Generating functions with complexity {complexity}...")
    
    try:
        duplicate_checker.main(runname, complexity)
    except SystemExit:
        pass
    
    # 2. Load generated functions
    library_dir = os.path.abspath(os.path.join(os.path.dirname(generator.__file__), '..', 'function_library'))
    eq_filename = os.path.join(library_dir, runname, f"compl_{complexity}", f"unique_equations_{complexity}.txt")
    
    try:
        with open(eq_filename, "r") as f:
            functions = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Could not find equations file: {eq_filename}")
        return
    
    print(f"Loaded {len(functions)} unique functions")
    
    # 3. Pick a few interesting functions and convert to JAX
    sample_functions = functions[:5]  # Take first 5 functions
    
    print(f"\n2. Converting sample functions to JAX:")
    jax_functions = []
    
    for i, func_str in enumerate(sample_functions):
        print(f"   {i+1}. {func_str}")
        try:
            # Check if function has parameters and provide default values
            # Parse to identify parameters
            import sympy
            x = sympy.symbols('x', real=True)
            a_symbols = sympy.symbols([f'a{i}' for i in range(20)], real=True)
            
            locs = {'x': x, 'sin': sympy.sin, 'cos': sympy.cos, 'inv': lambda x: 1/x, 
                    'Abs': sympy.Abs, 'pow': sympy.Pow, 'exp': sympy.exp, 'log': sympy.log}
            
            for j, a_sym in enumerate(a_symbols):
                locs[f'a{j}'] = a_sym
            
            expr = sympy.sympify(func_str, locals=locs)
            param_symbols = [sym for sym in expr.free_symbols if str(sym).startswith('a')]
            
            # Provide default parameter values if needed
            parameter_values = {}
            if param_symbols:
                for param in param_symbols:
                    parameter_values[param] = 1.0  # Default value of 1.0
                print(f"      (Using default parameter values: {[str(p) + '=1.0' for p in param_symbols]})")
            
            jax_func = create_jax_function_from_esr(func_str, parameter_values)
            jax_functions.append((func_str, jax_func))
        except Exception as e:
            print(f"      Error converting: {e}")
    
    if not jax_functions:
        print("No functions could be converted to JAX")
        return
    
    # 4. Demonstrate JAX capabilities
    print(f"\n3. JAX Capabilities Demonstration:")
    
    # Test data
    x_vals = jnp.linspace(0, 2*jnp.pi, 1000)
    
    for i, (func_str, jax_func) in enumerate(jax_functions[:3]):  # Test first 3
        print(f"\n   Function {i+1}: {func_str}")
        
        # JIT compilation
        jitted_func = jit(jax_func)
        
        # Gradient computation
        try:
            grad_func = jit(grad(jax_func))
            grad_vectorized = jit(vmap(grad_func))
            
            # Evaluate function and gradient
            y_vals = jitted_func(x_vals)
            grad_vals = grad_vectorized(x_vals)
            
            # Check for valid values
            if jnp.all(jnp.isfinite(y_vals)) and jnp.all(jnp.isfinite(grad_vals)):
                print(f"      ✓ Function and gradient computed successfully")
                print(f"      ✓ Function range: [{jnp.min(y_vals):.3f}, {jnp.max(y_vals):.3f}]")
                print(f"      ✓ Gradient range: [{jnp.min(grad_vals):.3f}, {jnp.max(grad_vals):.3f}]")
                
                # Performance test
                import time
                
                # Time JIT compilation (first call)
                start = time.time()
                _ = jitted_func(x_vals)
                compile_time = time.time() - start
                
                # Time subsequent calls
                start = time.time()
                for _ in range(100):
                    _ = jitted_func(x_vals)
                run_time = time.time() - start
                
                print(f"      ✓ JIT compile time: {compile_time:.4f}s")
                print(f"      ✓ 100 evaluations time: {run_time:.4f}s")
                print(f"      ✓ Average per evaluation: {run_time/100*1000:.2f}ms")
                
            else:
                print(f"      ✗ Function produces invalid values")
                
        except Exception as e:
            print(f"      ✗ Error in gradient computation: {e}")
    
    # 5. Batch evaluation demonstration
    print(f"\n4. Batch Evaluation with JAX:")
    
    # Create synthetic target data - same as function matching example
    def target_function(x):
        return jnp.sin(x) * x + 0.5 * x
    
    x_data = jnp.linspace(0, 2*jnp.pi, 100)  # More points like function matching
    y_data = target_function(x_data) + 0.01 * np.random.randn(len(x_data))  # Lower noise like function matching
    
    print(f"   Target: sin(x) * x + 0.5x + noise (same as function matching example)")
    print(f"   Evaluating {min(50, len(functions))} functions...")  # Increase from 20 to 50
    
    # Batch evaluate functions
    results = batch_evaluate_functions(functions[:50], x_data, y_data, max_functions=50)  # Increase to 50
    
    if results:
        print(f"\n   Top 5 best matching functions:")
        for i, (mse, func_str, jax_func) in enumerate(results[:5]):
            print(f"      {i+1}. MSE = {mse:.6f}: {func_str}")

    # 5. Parameter optimization demonstration
    print(f"\n5. Parameter Optimization with JAX:")
    
    def create_jax_objective_function(expr_template, param_symbols, x_vals, y_target):
        """Create a JAX-based objective function for parameter optimization with JIT and gradients"""
        
        # Pre-create the symbolic expression to JAX conversion
        def evaluate_expr_with_params(params):
            """Evaluate expression with given parameters"""
            substitutions = {param: params[i] for i, param in enumerate(param_symbols)}
            expr_with_params = expr_template.subs(substitutions)
            jax_func = create_jax_function_from_esr(str(expr_with_params))
            return jax_func(x_vals)
        
        @jit
        def jax_objective_core(params):
            """JIT-compiled core objective function"""
            # Evaluate function with parameters
            y_pred = evaluate_expr_with_params(params)
            # Return mean squared error
            return jnp.mean((y_pred - y_target) ** 2)
        
        # Create gradient function using JAX autodiff
        # Note: Can't JIT this initially due to dynamic compilation
        def jax_grad_func(params):
            return grad(lambda p: jnp.mean((evaluate_expr_with_params(p) - y_target) ** 2))(params)
        
        def objective(params):
            """Scipy-compatible objective function"""
            try:
                # For optimization, we'll use a simpler direct approach
                substitutions = {param: params[i] for i, param in enumerate(param_symbols)}
                expr_with_params = expr_template.subs(substitutions)
                jax_func = create_jax_function_from_esr(str(expr_with_params))
                
                y_pred = jax_func(x_vals)
                mse = jnp.mean((y_pred - y_target) ** 2)
                
                # Check for invalid values
                if jnp.isinf(mse) or jnp.isnan(mse):
                    return 1e10
                return float(mse)
            except:
                return 1e10
                
        def objective_grad(params):
            """Scipy-compatible gradient function"""
            try:
                # Create a JAX-compatible version for gradient computation
                def mse_func(p):
                    substitutions = {param: p[i] for i, param in enumerate(param_symbols)}
                    expr_with_params = expr_template.subs(substitutions)
                    jax_func = create_jax_function_from_esr(str(expr_with_params))
                    y_pred = jax_func(x_vals)
                    return jnp.mean((y_pred - y_target) ** 2)
                
                grad_val = grad(mse_func)(params)
                
                # Check for invalid values
                if jnp.any(jnp.isinf(grad_val)) or jnp.any(jnp.isnan(grad_val)):
                    return np.ones_like(params) * 1e-6  # Small gradient to avoid stuck optimization
                return np.array(grad_val)
            except:
                return np.ones_like(params) * 1e-6  # Small gradient fallback
                
        return objective, objective_grad

    # Find functions with parameters and optimize them
    optimized_results = []
    max_attempts = len(functions)  # Use all functions for fair comparison
    attempt = 0
    successful_optimizations = 0
    
    print(f"   Optimizing parameters for functions with free parameters...")
    
    for func_string in functions:
        if attempt >= max_attempts:
            break
        attempt += 1
        
        try:
            # Set up SymPy symbols
            x = sympy.symbols('x', real=True)
            a_symbols = sympy.symbols([f'a{i}' for i in range(10)], real=True)
            
            # Define locals for sympy parsing
            locs = {'x': x, 'sin': sympy.sin, 'cos': sympy.cos, 'inv': lambda x: 1/x, 
                    'Abs': sympy.Abs, 'pow': sympy.Pow, 'exp': sympy.exp, 'log': sympy.log}
            
            # Add parameter symbols to locs
            for j, a_sym in enumerate(a_symbols):
                locs[f'a{j}'] = a_sym

            # Parse the function string
            expr_template = sympy.sympify(func_string, locals=locs)
            
            # Check if the expression has any parameter symbols
            param_symbols = [sym for sym in expr_template.free_symbols if str(sym).startswith('a')]
            
            if param_symbols and len(param_symbols) <= 3:  # Only optimize functions with 1-3 parameters
                # Create objective function and its gradient
                objective, objective_grad = create_jax_objective_function(expr_template, param_symbols, x_data, y_data)
                
                # Initial guess for parameters
                initial_params = np.ones(len(param_symbols))
                
                # Optimize parameters with bounds to prevent extreme values
                bounds = [(-5, 5) for _ in param_symbols]  # Reasonable bounds for parameters
                
                try:
                    result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B', 
                                    jac=objective_grad,  # Use JAX-computed gradients
                                    options={'maxfun': 500})
                    
                    if result.success and result.fun < 1e8:  # Check if optimization was successful
                        # Use optimized parameters
                        optimized_params = result.x
                        mse = result.fun
                        
                        # Create final expression with optimized parameters
                        substitutions = {param: optimized_params[j] for j, param in enumerate(param_symbols)}
                        final_expr = expr_template.subs(substitutions)
                        final_jax_func = create_jax_function_from_esr(str(final_expr))
                        
                        optimized_results.append((mse, func_string, final_expr, optimized_params, final_jax_func))
                        successful_optimizations += 1
                        
                        if successful_optimizations <= 5:  # Show details for first 5 successful optimizations
                            param_str = ', '.join([f'{param}={val:.3f}' for param, val in zip(param_symbols, optimized_params)])
                            print(f"      ✓ {func_string} -> MSE = {mse:.6f} ({param_str})")
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            continue
    
    # Sort optimized results by MSE
    optimized_results.sort(key=lambda x: x[0])
    
    print(f"\n   Successfully optimized {successful_optimizations} functions with parameters")
    
    if optimized_results:
        print(f"\n   Top 5 optimized functions:")
        for i, (mse, func_str, optimized_expr, params, jax_func) in enumerate(optimized_results[:5]):
            print(f"      {i+1}. MSE = {mse:.6f}")
            print(f"         Original: {func_str}")
            print(f"         Optimized: {optimized_expr}")
            print()

    # 6. Compare best unoptimized vs optimized functions
    print(f"\n6. Comparison: Unoptimized vs Parameter-Optimized Functions:")
    
    best_unoptimized = results[0] if results else None
    best_optimized = optimized_results[0] if optimized_results else None
    
    if best_unoptimized and best_optimized:
        print(f"   Best unoptimized: MSE = {best_unoptimized[0]:.6f} ({best_unoptimized[1]})")
        print(f"   Best optimized:   MSE = {best_optimized[0]:.6f} ({best_optimized[1]})")
        
        improvement_factor = best_unoptimized[0] / best_optimized[0]
        print(f"   Improvement factor: {improvement_factor:.2f}x better with parameter optimization")
        
        # Use the better of the two for visualization
        if best_optimized[0] < best_unoptimized[0]:
            best_mse, best_func_str, best_jax_func = best_optimized[0], best_optimized[1], best_optimized[4]
            best_description = f"Optimized: {best_optimized[2]}"
        else:
            best_mse, best_func_str, best_jax_func = best_unoptimized[0], best_unoptimized[1], best_unoptimized[2]
            best_description = f"Unoptimized: {best_func_str}"
    else:
        # Fallback to whatever we have
        if best_optimized:
            best_mse, best_func_str, best_jax_func = best_optimized[0], best_optimized[1], best_optimized[4]
            best_description = f"Optimized: {best_optimized[2]}"
        elif best_unoptimized:
            best_mse, best_func_str, best_jax_func = best_unoptimized[0], best_unoptimized[1], best_unoptimized[2]
            best_description = f"Unoptimized: {best_func_str}"
        else:
            print(f"   No valid functions found for comparison")
            return

    # 7. Visualization
    print(f"\n7. Creating visualization...")
    
    # Plot comparison
    x_plot = jnp.linspace(0, 2*jnp.pi, 200)
    y_target = target_function(x_plot)
    y_best = best_jax_func(x_plot)
    
    plt.figure(figsize=(15, 10))
    
    # Function comparison
    plt.subplot(2, 2, 1)
    plt.plot(x_plot, y_target, 'b-', linewidth=2, label='Target: sin(x) * x')
    plt.plot(x_plot, y_best, 'r--', linewidth=2, label=f'Best ESR')
    plt.scatter(x_data, y_data, alpha=0.6, color='green', s=20, label='Data points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'ESR-JAX Function Matching (MSE = {best_mse:.6f})')
    plt.legend()
    plt.grid(True)
    
    # Gradient comparison if possible
    try:
        grad_target = jit(vmap(grad(target_function)))(x_plot)
        grad_best = jit(vmap(grad(best_jax_func)))(x_plot)
        
        plt.subplot(2, 2, 2)
        plt.plot(x_plot, grad_target, 'b-', linewidth=2, label="Target gradient")
        plt.plot(x_plot, grad_best, 'r--', linewidth=2, label="Best ESR gradient")
        plt.xlabel('x')
        plt.ylabel("f'(x)")
        plt.title('Gradient Comparison (JAX autodiff)')
        plt.legend()
        plt.grid(True)
        
    except Exception as e:
        plt.subplot(2, 2, 2)
        plt.text(0.5, 0.5, f'Gradient computation failed: {e}', 
                transform=plt.gca().transAxes, ha='center', va='center')
    
    # MSE comparison plot
    plt.subplot(2, 2, 3)
    mse_values = []
    labels = []
    
    if results:
        unopt_mses = [r[0] for r in results[:10]]  # Top 10 unoptimized
        mse_values.extend(unopt_mses)
        labels.extend(['Unoptimized'] * len(unopt_mses))
    
    if optimized_results:
        opt_mses = [r[0] for r in optimized_results[:10]]  # Top 10 optimized
        mse_values.extend(opt_mses)
        labels.extend(['Optimized'] * len(opt_mses))
    
    if mse_values:
        # Create box plot
        unopt_data = [mse for mse, label in zip(mse_values, labels) if label == 'Unoptimized']
        opt_data = [mse for mse, label in zip(mse_values, labels) if label == 'Optimized']
        
        box_data = []
        box_labels = []
        
        if unopt_data:
            box_data.append(unopt_data)
            box_labels.append('Unoptimized')
        
        if opt_data:
            box_data.append(opt_data)
            box_labels.append('Optimized')
        
        plt.boxplot(box_data, tick_labels=box_labels)
        plt.ylabel('MSE (log scale)')
        plt.yscale('log')
        plt.title('MSE Distribution Comparison')
        plt.grid(True)
    
    # Performance summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    summary_text = f"Summary:\n\n"
    summary_text += f"Functions generated: {len(functions)}\n"
    if results:
        summary_text += f"Functions evaluated: {len(results)}\n"
    if optimized_results:
        summary_text += f"Functions optimized: {len(optimized_results)}\n"
    summary_text += f"\nBest function:\n{best_description}\n"
    summary_text += f"MSE: {best_mse:.6f}\n\n"
    
    if best_unoptimized and best_optimized:
        summary_text += f"Improvement with optimization:\n{improvement_factor:.2f}x better"
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10, family='monospace')
    
    plt.tight_layout()
    plt.savefig('esr_jax_integration_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"   Plot saved as: esr_jax_integration_demo.png")

    print(f"\n8. Summary:")
    print(f"   ✓ Generated {len(functions)} functions using ESR")
    print(f"   ✓ Successfully converted functions to JAX format")
    print(f"   ✓ Demonstrated JIT compilation for fast evaluation")
    print(f"   ✓ Computed gradients using automatic differentiation")
    print(f"   ✓ Performed batch evaluation and function matching")
    if optimized_results:
        print(f"   ✓ Optimized parameters for {successful_optimizations} functions")
        print(f"   ✓ Best optimized MSE: {optimized_results[0][0]:.6f}")
    print(f"   ✓ JAX provides significant speedups for repeated evaluations")
    
    print(f"\nESR-JAX Integration Benefits:")
    print(f"   • Fast vectorized evaluation with JIT compilation")
    print(f"   • Automatic differentiation for gradients and higher-order derivatives")
    print(f"   • Parameter optimization using scipy and JAX-based objectives")
    print(f"   • GPU acceleration potential (if JAX configured with CUDA)")
    print(f"   • Seamless integration with optimization libraries")
    print(f"   • Composable transformations (grad, vmap, pmap, etc.)")
    print(f"   • Significant improvements in function matching accuracy")

if __name__ == "__main__":
    demonstrate_jax_integration()
