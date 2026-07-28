import glob
import os

import torch
import torch.cuda
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

# from torchsparse import __version__

version_file = open("./torchsparse/version.py")
version = version_file.read().split("'")[1]
print("torchsparse version:", version)

# Check if we're building for ROCm/HIP
is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
print(f"ROCm/HIP detected: {is_rocm}")

if (torch.cuda.is_available() and CUDA_HOME is not None) or (
    os.getenv("FORCE_CUDA", "0") == "1"
):
    device = "cuda"
    # Always use pybind_cuda.cu - PyTorch will hipify it for ROCm
    pybind_fn = "pybind_cuda.cu"
else:
    device = "cpu"
    pybind_fn = f"pybind_{device}.cpp"

# Files with CUDA-specific PTX inline assembly that don't work with HIP
# These use tensor core MMA ops and PTX cvta.to.shared instructions
HIP_EXCLUDED_FILES = [
    'implicit_gemm',  # All implicit GEMM files have PTX assembly
    'fetch_on_demand',  # May also have issues
]

sources = [os.path.join("torchsparse", "backend", pybind_fn)]
for fpath in glob.glob(os.path.join("torchsparse", "backend", "**", "*")):
    if fpath.endswith("_cpu.cpp") and device in ["cpu", "cuda"]:
        sources.append(fpath)
    elif device == "cuda":
        # Always use _cuda.cu files - PyTorch will hipify them on ROCm
        # Our custom HipFixBuildExtension patches the hipified output
        if fpath.endswith("_cuda.cu"):
            # Exclude files with PTX assembly issues for ROCm
            if is_rocm and any(excluded in fpath for excluded in HIP_EXCLUDED_FILES):
                print(f"  Excluding (PTX assembly): {fpath}")
                continue
            sources.append(fpath)


print(f"Building with device: {device}, sources: {len(sources)} files")
if is_rocm:
    print("Excluded for HIP:", HIP_EXCLUDED_FILES)

extension_type = CUDAExtension if device == "cuda" else CppExtension
extra_compile_args = {
    "cxx": ["-g", "-O3", "-fopenmp", "-lgomp"],
    "nvcc": ["-O3", "-std=c++17"],
}

# Custom BuildExtension that fixes hipified files before compilation
import re

# HIP-compatible half2 atomicAdd implementation to inject into hipified files
HIP_HALF2_ATOMICADD = '''
// HIP-compatible atomicAdd for half2 - injected by torchsparse setup.py
//
// Newer ROCm provides this natively, as
//   inline __device__ __half2 atomicAdd(__half2* const, const __half2)
// in hip/amd_detail/amd_hip_fp16.h, guarded only by __clang__ && __HIP__ with no
// feature macro to test. Defining the shim as well is a hard "redefinition of
// 'atomicAdd'" error, so gate it on the HIP version.
//
// Verified against the ROCm/clr sources: absent in 6.4.2, 7.0.0 and 7.2.1;
// present in 7.14. The 7.3 cutoff below is therefore correct at both ends of the
// range that was checked, but the exact release that introduced it was not
// pinned down — if a ROCm between 7.3 and 7.13 turns out to lack it, widen this.
#if defined(__HIP_PLATFORM_AMD__) && (!defined(HIP_VERSION) || HIP_VERSION < 70300000)
__device__ __forceinline__ half2 atomicAdd(half2* address, half2 val) {
    // Fallback implementation using float atomics
    // Convert half2 to two separate half values, do atomic add via unsigned int CAS
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint;
    unsigned int assumed;
    do {
        assumed = old;
        half2 old_val = *reinterpret_cast<half2*>(&assumed);
        half2 new_val = __hadd2(old_val, val);
        old = atomicCAS(address_as_uint, assumed, *reinterpret_cast<unsigned int*>(&new_val));
    } while (assumed != old);
    return *reinterpret_cast<half2*>(&old);
}
#endif
'''

def fix_hipify_kernel_launches(content):
    """Fix malformed hipLaunchKernelGGL calls and add HIP compatibility shims.
    
    PyTorch's hipify converts CUDA <<<grid, block>>> to:
        hipLaunchKernelGGL(kernel, grid, block, 0, 0, 0, 0, args...)
    
    The correct format is:
        hipLaunchKernelGGL(kernel, grid, block, 0, 0, args...)
    
    This removes the extra "0, 0, " arguments.
    
    Also adds HIP-compatible atomicAdd for half2 types since HIP doesn't natively
    support atomicAdd for half2.
    """
    # Pattern: ", 0, 0, 0, 0," should be ", 0, 0,"
    fixed = re.sub(r',\s*0,\s*0,\s*0,\s*0,', ', 0, 0,', content)
    
    # Inject half2 atomicAdd if file uses atomicAdd with half2 and doesn't already have fix
    if 'atomicAdd' in fixed and 'half2' in fixed and 'HIP-compatible atomicAdd for half2' not in fixed:
        # Find the first include statement and add our shim after includes
        # Look for the last #include line before the first function/kernel
        lines = fixed.split('\n')
        last_include_idx = 0
        for idx, line in enumerate(lines):
            if line.strip().startswith('#include'):
                last_include_idx = idx
        # Insert our shim after the last include
        lines.insert(last_include_idx + 1, HIP_HALF2_ATOMICADD)
        fixed = '\n'.join(lines)
    
    return fixed


class HipFixBuildExtension(BuildExtension):
    """Custom build extension that fixes hipified files before compilation."""
    
    def build_extensions(self):
        if is_rocm:
            # Find and fix all hipified files before compilation
            import glob as glob_module
            hip_files = glob_module.glob("torchsparse/backend/**/*_hip.hip", recursive=True)
            for hip_file in hip_files:
                try:
                    with open(hip_file, 'r') as f:
                        content = f.read()
                    fixed = fix_hipify_kernel_launches(content)
                    if fixed != content:
                        with open(hip_file, 'w') as f:
                            f.write(fixed)
                        print(f"  Fixed hipLaunchKernelGGL in: {hip_file}")
                except Exception as e:
                    print(f"  Warning: Could not fix {hip_file}: {e}")
        
        super().build_extensions()

setup(
    name="torchsparse",
    version=version,
    packages=find_packages(),
    ext_modules=[
        extension_type(
            "torchsparse.backend", sources, extra_compile_args=extra_compile_args
        )
    ],
    url="https://github.com/mit-han-lab/torchsparse",
    install_requires=[
        "numpy",
        "backports.cached_property",
        "tqdm",
        "typing-extensions",
        "wheel",
        "rootpath",
        "torch",
        "torchvision"
    ],
    dependency_links=[
        'https://download.pytorch.org/whl/cu118'
    ],
    cmdclass={"build_ext": HipFixBuildExtension},
    zip_safe=False,
)

