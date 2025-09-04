import numpy as np
import sympy
import matplotlib.pyplot as plt
import esr.generation.duplicate_checker as duplicate_checker
import os
import random
from scipy.optimize import minimize

# Set random seed for reproducible comparison
np.random.seed(42)
random.seed(42)

# 1. Set up parameters for function generation
complexity = 6
# Use a predefined runname from the ESR library that includes sine functions
runname = "osc_maths"

print(f"Generating functions with complexity {complexity} for run '{runname}'...")

# 2. Generate and process equations using the main entry point from the library
try:
    duplicate_checker.main(runname, complexity)
except SystemExit:
    # The duplicate_checker.main function calls quit(), so we catch the SystemExit
    pass

print("\nFunction generation and processing complete.")

# 3. Read the generated functions from the file
# The files are saved in the ESR library's function_library directory
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
    return np.sin(x)*x + 0.5*x

# Create evaluation points in the interval [0, 2π]
x_eval = np.linspace(0, 2*np.pi, 100)
y_true = true_function(x_eval) + np.random.normal(0, 0.01, size=x_eval.shape)  # Add some noise

print(f"\nTarget function: y = x * sin(x) + 0.5x")
print(f"Evaluating functions over interval [0, 2π] with {len(x_eval)} points")

# 5. Evaluate all functions and find the best match with optimized parameters
best_function = None
best_error = float('inf')
best_callable = None
best_expr = None
best_params = None
valid_functions = 0
errors = []
function_results = []  # Store (error, function_string, optimized_expr) tuples

print(f"\nEvaluating {len(all_functions)} functions with parameter optimization...")

def create_objective_function(expr_template, param_symbols, x_vals, y_target):
    """Create an objective function for parameter optimization"""
    def objective(params):
        try:
            # Substitute parameters into expression
            substitutions = {param: params[i] for i, param in enumerate(param_symbols)}
            expr_with_params = expr_template.subs(substitutions)
            
            # Convert to callable and evaluate
            callable_func = sympy.lambdify([sympy.Symbol('x')], expr_with_params, modules=['numpy'])
            y_pred = callable_func(x_vals)
            
            # Check for invalid values
            if np.any(np.isinf(y_pred)) or np.any(np.isnan(y_pred)):
                return 1e10  # Large penalty for invalid values
                
            # Return mean squared error
            return np.mean((y_pred - y_target) ** 2)
        except:
            return 1e10  # Large penalty for any errors
    return objective

for i, func_string in enumerate(all_functions):
    if (i + 1) % 25 == 0:
        print(f"Processed {i + 1}/{len(all_functions)} functions...")
    
    try:
        # 5.1. Convert the string to a sympy expression
        x = sympy.symbols('x', real=True)
        # Create parameter symbols a0, a1, a2, etc.
        a_symbols = sympy.symbols([f'a{i}' for i in range(10)], real=True)
        
        # Define locals for sympy to understand the function string
        locs = {'x': x, 'sin': sympy.sin, 'cos': sympy.cos, 'inv': lambda x: 1/x, 
                'Abs': sympy.Abs, 'pow': sympy.Pow, 'exp': sympy.exp, 'log': sympy.log}
        
        # Add parameter symbols to locs
        for j, a_sym in enumerate(a_symbols):
            locs[f'a{j}'] = a_sym

        # Parse the function string
        expr_template = sympy.sympify(func_string, locals=locs)
        
        # Check if the expression has any parameter symbols
        param_symbols = [sym for sym in expr_template.free_symbols if str(sym).startswith('a')]
        
        if param_symbols:
            # 5.2. Optimize parameters
            # Initial guess for parameters
            initial_params = np.ones(len(param_symbols))
            
            # Create objective function
            objective = create_objective_function(expr_template, param_symbols, x_eval, y_true)
            
            # Optimize parameters with bounds to prevent extreme values
            bounds = [(-10, 10) for _ in param_symbols]  # Reasonable bounds for parameters
            
            try:
                result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B', 
                                options={'maxfun': 1000})
                
                if result.success:
                    # Use optimized parameters
                    optimized_params = result.x
                    mse = result.fun
                    
                    # Create final expression with optimized parameters
                    substitutions = {param: optimized_params[j] for j, param in enumerate(param_symbols)}
                    final_expr = expr_template.subs(substitutions)
                    final_callable = sympy.lambdify([x], final_expr, modules=['numpy'])
                    
                else:
                    # Optimization failed, skip this function
                    continue
                    
            except:
                # Optimization failed, skip this function
                continue
                
        else:
            # No parameters to optimize
            try:
                final_callable = sympy.lambdify([x], expr_template, modules=['numpy'])
                y_pred = final_callable(x_eval)
                
                # Check for invalid values
                if np.any(np.isinf(y_pred)) or np.any(np.isnan(y_pred)):
                    continue
                    
                mse = np.mean((y_pred - y_true) ** 2)
                final_expr = expr_template
                optimized_params = None
                
            except:
                continue
        
        # 5.3. Record results
        errors.append(mse)
        function_results.append((mse, func_string, final_expr))
        
        # 5.4. Check if this is the best function so far
        if mse < best_error:
            best_error = mse
            best_function = func_string
            best_callable = final_callable
            best_expr = final_expr
            best_params = optimized_params
            
        valid_functions += 1
            
    except Exception:
        # Function parsing failed
        continue

print(f"\nEvaluation complete!")
print(f"Successfully evaluated {valid_functions} out of {len(all_functions)} functions")

if best_function is None:
    print("No valid function found that could be evaluated.")
    exit()

print(f"\nBest matching function:")
print(f"Function string: {best_function}")
print(f"Optimized expression: {best_expr}")
if best_params is not None:
    print(f"Optimized parameters: {[f'{val:.4f}' for val in best_params]}")
else:
    print("No parameters to optimize (function has no free parameters)")
print(f"Mean Squared Error: {best_error:.6f}")

# 6. Plot the results
x_plot = x_eval #np.linspace(0, 2*np.pi, 200)
y_true_plot = y_true # true_function(x_plot)
y_best_plot = best_callable(x_plot)

plt.figure(figsize=(12, 8))

# Main plot
plt.subplot(2, 1, 1)
plt.plot(x_plot, y_true_plot, 'b-', linewidth=2, label='True function: x * sin(x) + 0.5x')
plt.plot(x_plot, y_best_plot, 'r--', linewidth=2, label=f'Best ESR match: {best_function}')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Function Matching: ESR vs True Function (MSE = {best_error:.6f})')
plt.legend()
plt.grid(True)
plt.xlim(0, 2*np.pi)

# Error plot
plt.subplot(2, 1, 2)
error_plot = np.abs(y_true_plot - y_best_plot)
plt.plot(x_plot, error_plot, 'g-', linewidth=1.5, label='Absolute Error')
plt.xlabel('x')
plt.ylabel('|True - Predicted|')
plt.title('Absolute Error Between True and Best ESR Function')
plt.legend()
plt.grid(True)
plt.xlim(0, 2*np.pi)

plt.tight_layout()
plot_filename = 'function_matching_example.png'
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.show()

print(f"\nPlot saved as {plot_filename}")

# 7. Show statistics about all evaluated functions
if len(errors) > 0:
    errors = np.array(errors)
    print(f"\nStatistics for all {len(errors)} valid functions:")
    print(f"Best (lowest) MSE: {np.min(errors):.6f}")
    print(f"Worst (highest) MSE: {np.max(errors):.6f}")
    print(f"Mean MSE: {np.mean(errors):.6f}")
    print(f"Median MSE: {np.median(errors):.6f}")
    print(f"Standard deviation of MSE: {np.std(errors):.6f}")
    
    # Show top 5 best functions using the properly tracked results
    sorted_results = sorted(function_results, key=lambda x: x[0])  # Sort by MSE
    print(f"\nTop 5 best matching functions:")
    for i in range(min(5, len(sorted_results))):
        mse, func_string, optimized_expr = sorted_results[i]
        print(f"{i+1}. MSE = {mse:.6f}")
        print(f"   Original: {func_string}")
        print(f"   Optimized: {optimized_expr}")
        print()

print(f"\nExample complete. Generated equations are stored in: {library_dir}/{runname}/compl_{complexity}/")
