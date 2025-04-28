from graphviz import Source

# Path to the .gv.txt file
#input_path = 'project_overview_desc.gv.txt'  # Replace with your actual file path
#output_path = 'project_overview_desc'

#input_path = 'ml_workflow_overview_v2.gv.txt'  # Replace with your actual file path
#output_path = 'ml_workflow_overview_v2'

input_path = 'project_overview_outline.gv.txt'  # Replace with your actual file path
output_path = 'project_overview_outline'


# Choose your format: 'pdf', 'png', 'svg', etc.
output_format = 'pdf'

# Load the DOT source from file
with open(input_path, 'r') as f:
    dot_code = f.read()

# Create a Source object from the DOT content
graph = Source(dot_code, format=output_format)

# Render to file
graph.render(output_path, view=True)  # Set view=False if you don’t want it to auto-open

print(f"Rendered file saved as {output_path}.{output_format}")
