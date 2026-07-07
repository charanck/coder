"""Comprehensive tests for tree-sitter symbol extraction of all symbol types."""

import tempfile
from pathlib import Path


from core.service.tree_sitter import TreeSitterService


class TestExtractFunctions:
    """Tests for function extraction."""

    def test_extract_simple_function(self):
        """Test extracting a simple function."""
        code = """def hello():
    pass
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name

        try:
            file_uri = Path(temp_file).resolve().as_uri()
            service = TreeSitterService(".py")
            symbols = service.extract_document_symbols(file_uri)

            assert len(symbols) == 1
            assert symbols[0]['name'] == 'hello'
            assert symbols[0]['kind'] == 'Function'
        finally:
            Path(temp_file).unlink()

    def test_extract_async_function(self):
        """Test extracting an async function."""
        code = """async def fetch_data():
    pass
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name

        try:
            file_uri = Path(temp_file).resolve().as_uri()
            service = TreeSitterService(".py")
            symbols = service.extract_document_symbols(file_uri)

            assert len(symbols) == 1
            assert symbols[0]['name'] == 'fetch_data'
            assert symbols[0]['kind'] == 'Function'
        finally:
            Path(temp_file).unlink()

    def test_extract_function_with_parameters(self):
        """Test extracting function with parameters and type hints."""
        code = """def process(data: list, count: int = 0) -> str:
    return ""
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name

        try:
            file_uri = Path(temp_file).resolve().as_uri()
            service = TreeSitterService(".py")
            symbols = service.extract_document_symbols(file_uri)

            assert len(symbols) == 1
            assert symbols[0]['name'] == 'process'
            assert symbols[0]['kind'] == 'Function'
        finally:
            Path(temp_file).unlink()


class TestExtractClasses:
    """Tests for class extraction."""

    def test_extract_simple_class(self):
        """Test extracting a simple class."""
        code = """class MyClass:
    pass
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name

        try:
            file_uri = Path(temp_file).resolve().as_uri()
            service = TreeSitterService(".py")
            symbols = service.extract_document_symbols(file_uri)

            assert len(symbols) == 1
            assert symbols[0]['name'] == 'MyClass'
            assert symbols[0]['kind'] == 'Class'
        finally:
            Path(temp_file).unlink()

    def test_extract_class_with_methods(self):
        """Test extracting class with methods."""
        code = """class DataHandler:
    def __init__(self):
        pass
    
    def process(self):
        pass
    
    def clean(self):
        pass
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name

        try:
            file_uri = Path(temp_file).resolve().as_uri()
            service = TreeSitterService(".py")
            symbols = service.extract_document_symbols(file_uri)

            # Should extract class and all methods
            assert len(symbols) >= 4
            names = [s['name'] for s in symbols]
            assert 'DataHandler' in names
            assert '__init__' in names
            assert 'process' in names
            assert 'clean' in names
        finally:
            Path(temp_file).unlink()

    def test_extract_nested_classes(self):
        """Test extracting nested classes."""
        code = """class Outer:
    class Inner:
        pass
    
    def method(self):
        pass
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name

        try:
            file_uri = Path(temp_file).resolve().as_uri()
            service = TreeSitterService(".py")
            symbols = service.extract_document_symbols(file_uri)

            names = [s['name'] for s in symbols]
            assert 'Outer' in names
            assert 'Inner' in names
        finally:
            Path(temp_file).unlink()


class TestExtractVariables:
    """Tests for variable extraction."""

    def test_extract_simple_variable(self):
        """Test extracting a simple variable assignment."""
        code = """counter = 0
name = "test"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name

        try:
            file_uri = Path(temp_file).resolve().as_uri()
            service = TreeSitterService(".py")
            symbols = service.extract_document_symbols(file_uri)

            names = [s['name'] for s in symbols]
            assert 'counter' in names
            assert 'name' in names
            
            # Check kinds
            for sym in symbols:
                if sym['name'] in ('counter', 'name'):
                    assert sym['kind'] == 'Variable'
        finally:
            Path(temp_file).unlink()

    def test_extract_typed_variable(self):
        """Test extracting variables with type hints."""
        code = """count: int = 42
items: list[str] = []
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name

        try:
            file_uri = Path(temp_file).resolve().as_uri()
            service = TreeSitterService(".py")
            symbols = service.extract_document_symbols(file_uri)

            names = [s['name'] for s in symbols]
            assert 'count' in names
            assert 'items' in names
        finally:
            Path(temp_file).unlink()


class TestExtractConstants:
    """Tests for constant extraction."""

    def test_extract_constants(self):
        """Test extracting constants (uppercase variables)."""
        code = """MAX_SIZE = 100
DEBUG_MODE = True
MIN_VALUE = -999
CONFIG_PATH = "/etc/config"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name

        try:
            file_uri = Path(temp_file).resolve().as_uri()
            service = TreeSitterService(".py")
            symbols = service.extract_document_symbols(file_uri)

            constants = [s for s in symbols if s['kind'] == 'Constant']
            const_names = [s['name'] for s in constants]
            
            assert 'MAX_SIZE' in const_names
            assert 'DEBUG_MODE' in const_names
            assert 'MIN_VALUE' in const_names
            assert 'CONFIG_PATH' in const_names
        finally:
            Path(temp_file).unlink()

    def test_extract_mixed_variables_and_constants(self):
        """Test extracting both constants and variables in same file."""
        code = """# Constants
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30

# Variables
current_retries = 0
elapsed_time = 0
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name

        try:
            file_uri = Path(temp_file).resolve().as_uri()
            service = TreeSitterService(".py")
            symbols = service.extract_document_symbols(file_uri)

            constants = [s['name'] for s in symbols if s['kind'] == 'Constant']
            variables = [s['name'] for s in symbols if s['kind'] == 'Variable']
            
            assert 'MAX_RETRIES' in constants
            assert 'TIMEOUT_SECONDS' in constants
            assert 'current_retries' in variables
            assert 'elapsed_time' in variables
        finally:
            Path(temp_file).unlink()


class TestExtractImports:
    """Tests for module/import extraction."""

    def test_extract_from_import(self):
        """Test extracting from import statements."""
        code = """from pathlib import Path
from collections import defaultdict
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name

        try:
            file_uri = Path(temp_file).resolve().as_uri()
            service = TreeSitterService(".py")
            symbols = service.extract_document_symbols(file_uri)

            modules = [s['name'] for s in symbols if s['kind'] == 'Module']
            assert 'pathlib' in modules
            assert 'collections' in modules
        finally:
            Path(temp_file).unlink()

    def test_extract_from_import_with_multiple_items(self):
        """Test extracting from imports with multiple items."""
        code = """from typing import List, Dict, Optional
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name

        try:
            file_uri = Path(temp_file).resolve().as_uri()
            service = TreeSitterService(".py")
            symbols = service.extract_document_symbols(file_uri)

            modules = [s['name'] for s in symbols if s['kind'] == 'Module']
            assert 'typing' in modules
        finally:
            Path(temp_file).unlink()


class TestExtractDecoratedSymbols:
    """Tests for decorated functions and classes."""

    def test_extract_decorated_function(self):
        """Test extracting decorated function."""
        code = """@property
def value(self):
    return self._value

@staticmethod
def create():
    pass

@decorator
def process():
    pass
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name

        try:
            file_uri = Path(temp_file).resolve().as_uri()
            service = TreeSitterService(".py")
            symbols = service.extract_document_symbols(file_uri)

            names = [s['name'] for s in symbols]
            assert 'value' in names
            assert 'create' in names
            assert 'process' in names
        finally:
            Path(temp_file).unlink()

    def test_extract_decorated_class(self):
        """Test extracting decorated class."""
        code = """@dataclass
class Person:
    name: str
    age: int

@decorator
class Config:
    pass
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name

        try:
            file_uri = Path(temp_file).resolve().as_uri()
            service = TreeSitterService(".py")
            symbols = service.extract_document_symbols(file_uri)

            names = [s['name'] for s in symbols]
            assert 'Person' in names
            assert 'Config' in names
        finally:
            Path(temp_file).unlink()


class TestExtractComplex:
    """Tests for complex files with multiple symbol types."""

    def test_extract_all_symbol_types(self):
        """Test extracting all symbol types from a complex file."""
        code = """\"\"\"Module docstring.\"\"\"

from typing import List
from pathlib import Path

# Constants
MAX_SIZE = 100
DEBUG = True

# Variable
counter = 0

class DataProcessor:
    \"\"\"A data processor class.\"\"\"
    
    CONFIG_KEY = "processor"
    
    def __init__(self, name: str):
        self.name = name
    
    @property
    def status(self):
        return "active"
    
    def process(self, data: List[str]) -> int:
        return len(data)

async def fetch_data():
    pass

def main():
    processor = DataProcessor("main")
    processor.process(["a", "b"])

if __name__ == "__main__":
    main()
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name

        try:
            file_uri = Path(temp_file).resolve().as_uri()
            service = TreeSitterService(".py")
            symbols = service.extract_document_symbols(file_uri)

            # Verify we got all symbol types
            names = {s['name']: s['kind'] for s in symbols}
            
            # Functions
            assert 'fetch_data' in names and names['fetch_data'] == 'Function'
            assert 'main' in names and names['main'] == 'Function'
            
            # Classes
            assert 'DataProcessor' in names and names['DataProcessor'] == 'Class'
            
            # Methods
            assert '__init__' in names and names['__init__'] == 'Function'
            assert 'process' in names and names['process'] == 'Function'
            assert 'status' in names and names['status'] == 'Function'
            
            # Constants and Variables
            assert 'MAX_SIZE' in names and names['MAX_SIZE'] == 'Constant'
            assert 'DEBUG' in names and names['DEBUG'] == 'Constant'
            assert 'counter' in names and names['counter'] == 'Variable'
            assert 'CONFIG_KEY' in names and names['CONFIG_KEY'] == 'Constant'
            
            # Modules/Imports
            assert 'typing' in names and names['typing'] == 'Module'
            assert 'pathlib' in names and names['pathlib'] == 'Module'
        finally:
            Path(temp_file).unlink()

    def test_count_symbols(self):
        """Test that all symbols are extracted correctly."""
        code = """\"\"\"Module.\"\"\"

MAX_CONFIG = 10
config = None

def func1():
    pass

def func2():
    pass

class Class1:
    VAR = 1
    
    def method1(self):
        pass
    
    def method2(self):
        pass

class Class2:
    pass

from os import path
from sys import argv
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name

        try:
            file_uri = Path(temp_file).resolve().as_uri()
            service = TreeSitterService(".py")
            symbols = service.extract_document_symbols(file_uri)

            # Count by kind
            kind_counts = {}
            for sym in symbols:
                kind = sym['kind']
                kind_counts[kind] = kind_counts.get(kind, 0) + 1

            # Expected counts
            assert kind_counts.get('Function', 0) >= 4  # func1, func2, method1, method2
            assert kind_counts.get('Class', 0) >= 2  # Class1, Class2
            assert kind_counts.get('Constant', 0) >= 2  # MAX_CONFIG, VAR
            assert kind_counts.get('Variable', 0) >= 1  # config
            assert kind_counts.get('Module', 0) >= 2  # os, sys
        finally:
            Path(temp_file).unlink()


class TestCRLFLineEndings:
    """Tests for handling CRLF line endings correctly."""

    def test_extract_with_crlf_line_endings(self):
        """Test extraction works correctly with CRLF line endings."""
        code = "MAX_SIZE = 100\r\n\r\ndef hello():\r\n    pass\r\n\r\nclass MyClass:\r\n    pass\r\n"
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.py', delete=False) as f:
            f.write(code.encode('utf-8'))
            temp_file = f.name

        try:
            file_uri = Path(temp_file).resolve().as_uri()
            service = TreeSitterService(".py")
            symbols = service.extract_document_symbols(file_uri)

            names = [s['name'] for s in symbols]
            assert 'MAX_SIZE' in names
            assert 'hello' in names
            assert 'MyClass' in names
        finally:
            Path(temp_file).unlink()
