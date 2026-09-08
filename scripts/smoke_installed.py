"""Run outside the checkout after installing a Maturin wheel."""
import importlib.metadata
import pllm
import numpy as np
from pllm.native import MaskedGEMM, capabilities

assert importlib.metadata.version("pllm") == pllm.__version__
assert capabilities()["implementation"] == "rust"
w = np.array([[1,-2,3],[-7,0,7]],dtype=np.int8)
x = np.array([[3,9,65530]],dtype=np.uint32)
k = MaskedGEMM(threads=1)
assert k.native and k.library_path.is_file()
np.testing.assert_array_equal(k.compile(w).modular(x,65537), (x.astype(np.int64)@w.astype(np.int64).T)%65537)
print(f"Installed PLLM {pllm.__version__}: compiled Rust extension and arithmetic passed")
