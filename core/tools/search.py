import re

from langchain.tools import tool


@tool
def grep(pattern: str, file_path: str) -> list[str]:
    """Search for a pattern in a file and return matching lines.
    
    Args:
        pattern (str): The regex pattern to search for.
        file_path (str): The path to the file to be searched.

    Returns:
        list[str]: A list of lines that match the pattern or an error message if the file cannot be read.
    """
    try:
        with open(file_path,"r",encoding="utf-8") as f:
            lines = f.readlines()
        
        matches = [line for line in lines if re.search(pattern, line)]
        return matches
    except Exception as e:
        return [f"Error reading file: {e}"]