import os

# Define the project structure as a dictionary
# Keys are folder names (relative to base_dir), and values are lists of files to create inside.
project_structure = {
    "project": {
        "data_generation": [
            "__init__.py",
            "operators.py",
            "parameters.py",
            "mixing.py",
            "granulation.py",
            "drying.py",
            "compression.py",
            "simulator.py",
            "utils.py"
        ],
        "": [  # Files directly under "project" folder
            "main.py",
            "requirements.txt",
            "README.md"
        ]
    }
}

def create_file(file_path, content=""):
    """Create a file at file_path with optional content."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def create_structure(base_dir, structure):
    """
    Recursively create folders and files based on the provided structure.
    
    Parameters:
    - base_dir (str): The root directory where the structure will be created.
    - structure (dict): A dictionary where keys are directory names and values are either:
        - A list of file names (if there is no subdirectory), or
        - Another dict representing a nested structure.
    """
    for folder, contents in structure.items():
        # Build the full folder path
        folder_path = os.path.join(base_dir, folder) if folder else base_dir
        os.makedirs(folder_path, exist_ok=True)
        
        if isinstance(contents, dict):
            # Recursively create subdirectories
            create_structure(folder_path, contents)
        elif isinstance(contents, list):
            # Create each file in the folder
            for filename in contents:
                file_path = os.path.join(folder_path, filename)
                # Optionally add a header comment depending on file type
                if filename.endswith(".py"):
                    header = f'"""Module: {filename}"""\n\n'
                elif filename.endswith(".md"):
                    header = f"# {filename}\n\nProject documentation."
                elif filename.endswith(".txt"):
                    header = "# Requirements\n\n"
                else:
                    header = ""
                create_file(file_path, header)

if __name__ == "__main__":
    # Set the base directory where you want the project structure created.
    # For example, use the current working directory.
    base_directory = os.getcwd()  # or specify a custom path, e.g. "/path/to/base_dir"
    
    # Create the structure
    create_structure(base_directory, project_structure)
    
    print(f"Project structure created under: {os.path.join(base_directory, 'project')}")

