import sys
import importlib

from lmcache_vllm.vllm_injection import InitLMCacheEnvironment
from lmcache_vllm.vllm_adapter import close_lmcache_engine

from lmcache.logging import init_logger
logger = init_logger(__name__)

EXPECTED_VLLM_VERSION = "0.6.1.post2"
__version__ = EXPECTED_VLLM_VERSION

def check_library_version(library_name, required_version):
    try:
        # Dynamically import the specified library
        lib = importlib.import_module(library_name)
        # Check if the installed version matches the required version
        if hasattr(lib, '__version__'):
            if lib.__version__ == required_version:
                return True
            else:
                logger.error(f"Version mismatch: {lib.__version__} found, {required_version} required.")
                return False
        else:
            logger.error(f"The library {library_name} does not have a '__version__' attribute.")
            return False
    except ModuleNotFoundError:
        logger.error(f"Library {library_name} is not installed.")
        return False

def initialize_environment():
    # Check vllm and it's version
    assert check_library_version("vllm", EXPECTED_VLLM_VERSION), f"vllm {EXPECTED_VLLM_VERSION} not found"
    InitLMCacheEnvironment()

initialize_environment()

# Function to import and return modules dynamically
def dynamic_import(name):
    try:
        # Directly import and return the module from vllm
        return importlib.import_module(name)
    except ImportError:
        raise ImportError(f"No module named {name}")

# Creating a proxy module class
class ProxyModule:
    def __init__(self, fullname):
        self.fullname = fullname
        self.module = None
        self.loader = None

    def _get_actual_name(self, fullname):
        if fullname != self.fullname:
            raise ImportError(f"Cannot import {fullname} by a ProxyModule({self.fullname})")

        prefix = f"{__name__}.vllm"
        return fullname.replace(prefix, "vllm")

    def get_origin(self):
        actual_name = self._get_actual_name(self.fullname)
        spec = importlib.util.find_spec(actual_name)
        if spec is None:
            return None
        else:
            return spec.origin

    def create_module(self, spec):
        # This method is part of the loader protocol and is optional
        return None  # Use default module creation

    def exec_module(self, module):
        # This replaces load_module; it directly executes the module in its context
        if self.module is None:
            self.load_module(self.fullname)
            #actual_name = self._get_actual_name(self.fullname)
            #self.module = dynamic_import(actual_name)
        # Copy attributes to the module
        module.__dict__.update(self.module.__dict__)

    def load_module(self, fullname):
        if self.module is None:
            actual_name = self._get_actual_name(fullname)
            self.module = importlib.import_module(actual_name)
        sys.modules[fullname] = self.module
        return self.module

    def get_code(self, fullname):
        if self.module is None:
            self.load_module(fullname)

        actual_name = self._get_actual_name(fullname)
        spec = importlib.util.find_spec(actual_name)

        if spec is None:
            raise ImportError(f"ProxyModule({fullname}) cannot find the actual module {actual_name}")

        if self.loader is None and spec.origin:
            self.loader = importlib.machinery.SourceFileLoader(spec.name, spec.origin)

        if self.loader:
            return self.loader.get_code(spec.name)
        else:
            raise ImportError(f"ProxyModule({fullname}) could not load source for {actual_name}")

        # HACK: change the sys.argv[0] if needed
        if sys.argv[0] is None or sys.argv[0] == "-m":
            sys.argv[0] = spec.origin

# Setting up module finder and loader
class ModuleFinder:
    def find_spec(self, fullname, path, target=None):
        prefix = f"{__name__}.vllm"
        if fullname.startswith(prefix):
            proxy_module = ProxyModule(fullname)
            return importlib.machinery.ModuleSpec(fullname, proxy_module, origin = proxy_module.get_origin())
        return None

sys.meta_path.insert(0, ModuleFinder())

__all__ = [
    "close_lmcache_engine",
]
