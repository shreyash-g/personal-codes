import ast
import os
import sys
import importlib.util
import json
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# Standard library and common third-party packages to ignore
STANDARD_PACKAGES = {
    # Python Standard Library
    'abc', 'argparse', 'asyncio', 'collections', 'concurrent', 'contextlib',
    'copy', 'csv', 'datetime', 'decimal', 'enum', 'functools', 'glob',
    'hashlib', 'hmac', 'http', 'importlib', 'io', 'itertools', 'json',
    'logging', 'math', 'multiprocessing', 'operator', 'os', 'pathlib',
    'pickle', 're', 'shutil', 'socket', 'sqlite3', 'string', 'sys',
    'tempfile', 'threading', 'time', 'traceback', 'typing', 'unittest',
    'urllib', 'uuid', 'warnings', 'weakref', 'xml', 'zipfile',
    
    # Common Third-party Packages
    'boto3', 'botocore', 'celery', 'django', 'flask', 'jinja2', 'mongoengine',
    'numpy', 'pandas', 'pip', 'pymongo', 'pytest', 'redis', 'requests',
    'setuptools', 'sqlalchemy', 'yaml', 'cProfile', 'pstats'
}

@dataclass
class ImportInfo:
    full_path: str
    file_location: str
    type: str  # 'function', 'class', 'constant', or 'unknown'
    is_used: bool = False  # Whether this import is actually used in the code
    alias: Optional[str] = None  # The alias name if the import is aliased

@dataclass
class VariableInfo:
    name: str
    type: str
    line_number: int
    is_resolved: bool = True  # Whether we could determine the type

@dataclass
class FunctionCall:
    name: str
    context: Optional[str] = None  # The object the function is called on
    is_method: bool = False  # Whether this is a method call (obj.method()) vs function call (func())
    chain: List[str] = None  # The chain of method calls leading to this one
    object_type: Optional[str] = None  # The type of the base object in the chain

    def __post_init__(self):
        if self.chain is None:
            self.chain = []

    def get_chain_with_types(self) -> str:
        """Get the chain with type information."""
        if not self.chain:
            return "None"
        
        parts = []
        for i, item in enumerate(self.chain):
            if i == 0 and self.object_type:  # Only show type for the base object
                parts.append(f"{item} ({self.object_type})")
            else:
                parts.append(item)
        return " -> ".join(parts)

@dataclass

class CodeAnalysis:
    imports: List[ImportInfo]
    function_calls: List[FunctionCall]
    class_calls: List[str]
    variables: List[VariableInfo]

@dataclass
class AnalysisOutput:
    """Container for the complete analysis output."""
    imports: List[ImportInfo]
    variables: List[VariableInfo]
    function_calls: List[FunctionCall]
    class_calls: List[str]

class CodeAnalyzer:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        # Add project root to Python path
        if self.root_dir not in sys.path:
            sys.path.insert(0, self.root_dir)
        self.imports: List[ImportInfo] = []
        self.function_calls: List[FunctionCall] = []
        self.class_calls: List[str] = []
        self.variables: List[VariableInfo] = []
        self.type_map: Dict[str, str] = {}  # Maps class names to their full paths
        
    def is_standard_package(self, module_name: str) -> bool:
        """Check if a module is a standard package that should be ignored."""
        root_package = module_name.split('.')[0]
        return root_package in STANDARD_PACKAGES
        
    def find_module_path(self, module_name: str) -> Optional[str]:
        """Find the actual file path for a module using Python's import system."""
        # Skip standard packages
        if self.is_standard_package(module_name):
            return None
            
        try:
            # Try to find the module using Python's import system
            spec = importlib.util.find_spec(module_name)
            if spec and spec.origin:
                return spec.origin
        except (ImportError, ModuleNotFoundError):
            pass
            
        # If Python's import system fails, try to find it in our project
        parts = module_name.split('.')
        
        # Try both app and apps as root
        for root in ['app', 'apps']:
            # Try as a module first (module.py)
            module_path = os.path.join(self.root_dir, root, *parts[:-1], f"{parts[-1]}.py")
            if os.path.exists(module_path):
                return module_path
                
            # Try as a package (module/__init__.py)
            init_path = os.path.join(self.root_dir, root, *parts, '__init__.py')
            if os.path.exists(init_path):
                return init_path
                
            # Try with the other root
            other_root = 'apps' if root == 'app' else 'app'
            module_path = os.path.join(self.root_dir, other_root, *parts[:-1], f"{parts[-1]}.py")
            if os.path.exists(module_path):
                return module_path
            init_path = os.path.join(self.root_dir, other_root, *parts, '__init__.py')
            if os.path.exists(init_path):
                return init_path
        
        return None

    def find_definition(self, module_path: str, name: str, visited=None) -> Tuple[Optional[str], Optional[str]]:
        """Find where an object is actually defined by following imports."""
        if visited is None:
            visited = set()
        
        if (module_path, name) in visited:
            return None, None
        visited.add((module_path, name))
        
        # Skip standard packages
        if self.is_standard_package(module_path):
            return None, None
            
        # Find the module file
        file_path = self.find_module_path(module_path)
        if not file_path:
            return None, None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            # Look for direct definition
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == name:
                    return file_path, 'class'
                elif isinstance(node, ast.FunctionDef) and node.name == name:
                    return file_path, 'function'
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == name:
                            return file_path, 'constant'
            
            # Look for imports
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    for alias in node.names:
                        if alias.name == name:
                            # Found an import, follow it
                            next_file, obj_type = self.find_definition(node.module, name, visited)
                            if next_file:
                                return next_file, obj_type
                                
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
        
        return None, None

    def get_type_from_call(self, node: ast.Call) -> Optional[str]:
        """Get the type of a function call result."""
        if isinstance(node.func, ast.Name):
            # Direct function call
            return self.type_map.get(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            # Method call
            if isinstance(node.func.value, ast.Name):
                # obj.method()
                obj_name = node.func.value.id
                return self.type_map.get(obj_name)
            elif isinstance(node.func.value, ast.Attribute):
                # Handle method chains (obj.method1().method2())
                return self.get_type_from_call(ast.Call(
                    func=node.func.value,
                    args=[],
                    keywords=[]
                ))
        return None

    def get_full_chain(self, node: ast.Call) -> List[str]:
        """Get the full chain of method calls leading to this call."""
        chain = []
        current = node.func
        
        # Walk up the chain of attributes
        while isinstance(current, ast.Attribute):
            chain.append(current.attr)
            current = current.value
            
            # If we hit another Call, process it recursively
            if isinstance(current, ast.Call):
                parent_chain = self.get_full_chain(current)
                if parent_chain:  # Only extend if we got a parent chain
                    chain.extend(parent_chain)
                # Move to the function of the call
                current = current.func
                
        # Add the base object if it's a name
        if isinstance(current, ast.Name):
            chain.append(current.id)
            
        # Remove duplicates while preserving order
        seen = set()
        unique_chain = []
        for item in chain[::-1]:  # Process in reverse to keep the most recent occurrences
            if item not in seen:
                seen.add(item)
                unique_chain.append(item)
        return unique_chain  # Already in correct order

    def analyze(self, script_content: str) -> CodeAnalysis:
        """Analyze the Python script content and return import and call information."""
        try:
            tree = ast.parse(script_content)
            
            # First pass: collect all imports and build type map
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if not self.is_standard_package(name.name):
                            file_loc, obj_type = self.find_definition('', name.name)
                            self.imports.append(ImportInfo(
                                full_path=name.name,
                                file_location=file_loc or "Could not resolve",
                                type=obj_type or "unknown",
                                is_used=False,
                                alias=name.asname  # Store the alias if it exists
                            ))
                            if obj_type == 'class':
                                self.type_map[name.name] = name.name
                                if name.asname:  # Also map the alias to the same type
                                    self.type_map[name.asname] = name.name
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for name in node.names:
                        full_path = f"{module}.{name.name}"
                        if not self.is_standard_package(module):
                            file_loc, obj_type = self.find_definition(module, name.name)
                            self.imports.append(ImportInfo(
                                full_path=full_path,
                                file_location=file_loc or "Could not resolve",
                                type=obj_type or "unknown",
                                is_used=False,
                                alias=name.asname  # Store the alias if it exists
                            ))
                            if obj_type == 'class':
                                self.type_map[name.name] = full_path
                                if name.asname:  # Also map the alias to the same type
                                    self.type_map[name.asname] = full_path

            # Second pass: track import usage
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    # Check if this name matches any of our imports or their aliases
                    for imp in self.imports:
                        if imp.full_path.split('.')[-1] == node.id or imp.alias == node.id:
                            imp.is_used = True
                elif isinstance(node, ast.Attribute):
                    # Check for attribute access (e.g., module.function)
                    if isinstance(node.value, ast.Name):
                        for imp in self.imports:
                            parts = imp.full_path.split('.')
                            if len(parts) > 1:
                                # Check both original name and alias
                                if (parts[0] == node.value.id or imp.alias == node.value.id) and parts[-1] == node.attr:
                                    imp.is_used = True

            # Third pass: track variable assignments
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_type = None
                            is_resolved = False
                            
                            if isinstance(node.value, ast.Call):
                                # Handle class instantiation or method call
                                var_type = self.get_type_from_call(node.value)
                                is_resolved = bool(var_type)
                            elif isinstance(node.value, ast.Name):
                                # Variable assignment from another variable
                                var_type = self.type_map.get(node.value.id)
                                is_resolved = bool(var_type)
                            elif isinstance(node.value, ast.List):
                                var_type = "List"
                                is_resolved = True
                            elif isinstance(node.value, ast.Dict):
                                var_type = "Dict"
                                is_resolved = True
                            elif isinstance(node.value, ast.Constant):
                                var_type = type(node.value.value).__name__
                                is_resolved = True
                            
                            self.variables.append(VariableInfo(
                                name=target.id,
                                type=var_type or "unknown",
                                line_number=node.lineno,
                                is_resolved=is_resolved
                            ))
                            if var_type:
                                self.type_map[target.id] = var_type

            # Fourth pass: collect function calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        # Direct function call
                        self.function_calls.append(FunctionCall(
                            name=node.func.id,
                            is_method=False
                        ))
                    elif isinstance(node.func, ast.Attribute):
                        # Method call
                        chain = self.get_full_chain(node)
                        context = chain[0] if chain else None
                        
                        # Get type of the base object only
                        object_type = None
                        if chain:
                            object_type = self.type_map.get(chain[0])
                            
                        self.function_calls.append(FunctionCall(
                            name=node.func.attr,
                            context=context,
                            is_method=True,
                            chain=chain,
                            object_type=object_type
                        ))
                
                # Handle class instantiations
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.class_calls.append(node.func.id)
            
            return CodeAnalysis(
                imports=sorted(self.imports, key=lambda x: x.full_path),
                function_calls=sorted(self.function_calls, key=lambda x: x.name),
                class_calls=sorted(list(set(self.class_calls))),
                variables=sorted(self.variables, key=lambda x: x.line_number)
            )
            
        except Exception as e:
            print(f"Error analyzing script: {str(e)}")
            return CodeAnalysis([], [], [], [])

class ASTNodeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ast.AST):
            return {
                'type': obj.__class__.__name__,
                'fields': {k: v for k, v in ast.iter_fields(obj)},
                'lineno': getattr(obj, 'lineno', None),
                'col_offset': getattr(obj, 'col_offset', None)
            }
        return super().default(obj)

class EnhancedJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle our dataclasses."""
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            return asdict(obj)
        return super().default(obj)

def analyze_script_file(file_path: str) -> None:
    """Analyze a Python script file and print the results."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            script_content = f.read()
            
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(file_path)))
        analyzer = CodeAnalyzer(root_dir)
        results = analyzer.analyze(script_content)
        
        # Create structured output
        output = AnalysisOutput(
            imports=results.imports,
            variables=results.variables,
            function_calls=results.function_calls,
            class_calls=results.class_calls
        )
        
        # Print human-readable output
        print(f"\nAnalysis of {file_path}:")
        
        print("\nImports:")
        for imp in results.imports:
            print(f"  - {imp.full_path}")
            print(f"    Location: {imp.file_location}")
            print(f"    Type: {imp.type}")
            print(f"    Used: {'Yes' if imp.is_used else 'No'}")
            if imp.alias:
                print(f"    Alias: {imp.alias}")
        
        print("\nVariables:")
        print("  Resolved:")
        for var in results.variables:
            if var.is_resolved:
                print(f"  - {var.name}")
                print(f"    Type: {var.type}")
        
        print("\n  Unresolved:")
        for var in results.variables:
            if not var.is_resolved:
                print(f"  - {var.name}")
        
        print("\nFunction Calls:")
        for func in results.function_calls:
            if func.is_method:
                chain_str = func.get_chain_with_types()
                print(f"  - {func.name} (called on {chain_str})")
            else:
                print(f"  - {func.name}")
        
        print("\nClass Calls:")
        for cls in results.class_calls:
            print(f"  - {cls}")
            
        # Save JSON output
        json_output = json.dumps(output, cls=EnhancedJSONEncoder, indent=2)
        output_file = file_path + '.analysis.json'
        with open(output_file, 'w') as f:
            f.write(json_output)
        print(f"\nJSON output saved to: {output_file}")
            
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python code_analyzer.py <script_file>")
        sys.exit(1)
    
    analyze_script_file(sys.argv[1]) 
