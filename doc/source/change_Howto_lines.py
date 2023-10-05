# Specify the path to your input file
input_file = 'How_to_2DIR_spectra.rst'

# Define the new content for the first two lines
new_lines = [
    "Example\n",
    "=======\n"
]

# Read the content of the input file starting from the third line
with open(input_file, 'r') as file:
    remaining_lines = file.readlines()[2:]

# Write the new content followed by the remaining lines to the same file
with open(input_file, 'w') as file:
    file.writelines(new_lines + remaining_lines)
