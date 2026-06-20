from langchain.tools import tool
from pathlib import Path
import os
import shutil
import glob
from pydantic import BaseModel

@tool
def read_file(
    file_path: str,
    start_line: int | None = None,
    end_line: int | None = None
) -> str:
    """Read a file and return its content.
    
    Args:
        file_path (str): The path to the file to be read.
        start_line (int, optional): The line number to start reading from (1-indexed). Defaults to None.
        end_line (int, optional): The line number to stop reading at (1-indexed). Defaults to None.

    Returns:
        str: The content of the file or an error message if the file cannot be read.
    """
    try:
        with open(file_path,"r",encoding="utf-8") as f:
            lines = f.readlines()
        
        if start_line is not None and end_line is not None:
            if start_line < 1 or end_line > len(lines) or start_line > end_line:
                return "Invalid line range specified."
            return ''.join(lines[start_line - 1:end_line])
        
        return ''.join(lines)
    except Exception as e:
        return f"Error reading file: {e}"
    
@tool
def read_files(paths: list[str]) -> list[str]:
    """Read multiple files and return their contents.
    
    Args:
        paths (list[str]): A list of file paths to be read.
    Returns:
        list[str]: A list of file contents or error messages for each file.
    """
    contents = []
    for path in paths:
        content = read_file.run(path)
        contents.append(content)
    return contents


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file.
    
    Args:
        file_path (str): The path to the file to be written.
        content (str): The content to write to the file.

    Returns:
        str: A success message or an error message if the file cannot be written.
    """
    try:
        Path(file_path).parent.mkdir(parents=True,exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(content)
        return f"File '{file_path}' written successfully."
    except Exception as e:
        return f"Error writing file: {e}"
    
@tool
def edit_file(path: str,search: str,replace: str,replace_all: bool = False) -> str:
    """Edit a file by replacing a string with another string.
    
    Args:
        path (str): The path to the file to be edited.
        search (str): The string to search for in the file.
        replace (str): The string to replace the search string with.
        replace_all (bool, optional): Whether to replace all occurrences of the search string. Defaults to False.

    Returns:
        str: A success message or an error message if the file cannot be edited.
    """
    try:
        with open(path, 'r') as f:
            content = f.read()
        
        if search not in content:
            return f"Search string '{search}' not found in the file."
        
        if replace_all:
            content = content.replace(search, replace)
        else:
            content = content.replace(search, replace, 1)
        
        with open(path, 'w') as f:
            f.write(content)
        
        return f"File '{path}' edited successfully."
    except Exception as e:
        return f"Error editing file: {e}"
    
class Edit(BaseModel):
    search: str
    replace: str
    
@tool
def multi_edit_file(file_path: str, edits: list[Edit]) -> str:
    """Edit a file by applying multiple edits.
    
    Args:
        file_path (str): The path to the file to be edited.
        edits (list[Edit]): A list of Edit objects containing search and replace strings.

    Returns:
        str: A success message or an error message if the file cannot be edited.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        for edit in edits:
            if edit.search not in content:
                return f"Search string '{edit.search}' not found in the file."
            content = content.replace(edit.search, edit.replace)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        return f"File '{file_path}' edited successfully with multiple edits."
    except Exception as e:
        return f"Error editing file: {e}"

@tool
def list_files(directory_path: str, recursive: bool = False) -> str:
    """List all files in a directory.
    
    Args:
        directory_path (str): The path to the directory.
        recursive (bool, optional): Whether to list files recursively. Defaults to False.

    Returns:
        str: A list of files in the directory or an error message if the directory cannot be read.
    """
    try:
        if recursive:
            file_list = []
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_list.append(os.path.join(root, file))
            return '\n'.join(file_list)
        else:
            return '\n'.join(os.listdir(directory_path))
    except Exception as e:
        return f"Error listing files in directory: {e}"
    

@tool
def create_file(path: str, content : str="") -> str:
    """Create a new file with the specified content.
    
    Args:
        path (str): The path to the file to be created.
        content (str, optional): The content to write to the file. Defaults to an empty string.

    Returns:
        str: A success message or an error message if the file cannot be created.
    """
    try:
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        return f"File '{path}' created successfully."
    except Exception as e:
        return f"Error creating file: {e}"
    
@tool
def delete_file(path: str) -> str:
    """Delete a file.
    
    Args:
        path (str): The path to the file to be deleted.

    Returns:
        str: A success message or an error message if the file cannot be deleted.
    """
    try:
        os.remove(path)
        return f"File '{path}' deleted successfully."
    except Exception as e:
        return f"Error deleting file: {e}"
    
@tool
def move_file(source: str, destination: str) -> str:
    """Move a file from source to destination.
    
    Args:
        source (str): The path to the file to be moved.
        destination (str): The path to the destination where the file should be moved.

    Returns:
        str: A success message or an error message if the file cannot be moved.
    """
    try:
        Path(destination).parent.mkdir(parents=True,exist_ok=True)
        shutil.move(source, destination)
        return f"File moved from '{source}' to '{destination}' successfully."
    except Exception as e:
        return f"Error moving file: {e}"
    
@tool
def copy_file(source: str, destination: str) -> str:
    """Copy a file from source to destination.
    
    Args:
        source (str): The path to the file to be copied.
        destination (str): The path to the destination where the file should be copied.

    Returns:
        str: A success message or an error message if the file cannot be copied.
    """
    import shutil
    try:
        Path(destination).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy(source, destination)
        return f"File copied from '{source}' to '{destination}' successfully."
    except Exception as e:
        return f"Error copying file: {e}"
    
@tool
def find_files(pattern:str="*.go",root:str=".") -> str:
    """Find files matching a pattern in a directory.
    
    Args:
        pattern (str, optional): The pattern to match files against. Defaults to "*.go".
        root (str, optional): The root directory to start searching from. Defaults to ".".

    Returns:
        str: A list of files matching the pattern or an error message if the search fails.
    """

    try:
        search_pattern = os.path.join(root, '**', pattern)
        files = glob.glob(search_pattern, recursive=True)
        return '\n'.join(files)
    except Exception as e:
        return f"Error finding files: {e}"
    
@tool
def list_directories(directory_path: str, recursive: bool = False) -> str:
    """List all directories in a directory.
    
    Args:
        directory_path (str): The path to the directory.
        recursive (bool, optional): Whether to list directories recursively. Defaults to False.
    
    Returns:
        str: A list of directories in the directory or an error message if the directory cannot be read.
    """
    try:
        if recursive:
            dir_list = []
            for root, dirs, files in os.walk(directory_path):
                for dir in dirs:
                    dir_list.append(os.path.join(root, dir))
            return '\n'.join(dir_list)
        else:
            return '\n'.join([d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))])
    except Exception as e:
        return f"Error listing directories: {e}"
