
import numpy as np
import sympy
import matplotlib.pyplot as plt
import esr.generation.generator as generator

# 1. Define a function string and basis functions
func_string = "x**2 + 3"
basis_functions = [["x"], [], ["+", "pow"]]

# 2. Convert the string to a sympy expression
# Create sympy symbols for variables
x = sympy.symbols('x', real=True)
locs = {'x': x}

# Use the generator to get the sympy expression
expr, _, _ = generator.string_to_node(func_string, basis_functions, locs=locs)

print(f"Original string: '{func_string}'")
print(f"Sympy expression: {expr}")

# 3. Convert the sympy expression to a callable Python function
# The first argument to lambdify is a list of the independent variables
# The second argument is the expression to convert
callable_func = sympy.lambdify([x], expr)

print(f"Generated callable function: {callable_func}")

# 4. Test the callable function
x_vals = np.linspace(0, 10, 100)
y_vals = callable_func(x_vals)

# 5. Plot the results
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label=f'Callable function for "{func_string}"')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of a function generated from ESR')
plt.legend()
plt.grid(True)
plt.savefig('callable_example.png')
plt.show()

print("\nPlot saved as callable_example.png")
