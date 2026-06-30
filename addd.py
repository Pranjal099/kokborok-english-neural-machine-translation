# Get input from the user for the first number
num1_str = input("Enter the first number: ")

# Get input from the user for the second number
num2_str = input("Enter the second number: ")

# Convert the input strings to floating-point numbers
# This allows for both integer and decimal inputs
num1 = float(num1_str)
num2 = float(num2_str)

# Perform the addition
sum_result = num1 + num2

# Display the sum
print(f"The sum of {num1} and {num2} is {sum_result}")