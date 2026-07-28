import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

# AMD/ROCm: enable AOTriton experimental attention paths used by PyTorch SDPA.
# Must be set before `import torch`. Harmless on CUDA builds (only the ROCm
# path reads the flag).
os.environ.setdefault('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '1')

import torch

# Configure torchsparse for HIP/ROCm compatibility (mirrors app.py).
# Use GatherScatter dataflow instead of ImplicitGEMM (PTX-only).
if os.environ.get('SPARSE_BACKEND') == 'torchsparse' and hasattr(torch.version, 'hip'):
    try:
        from torchsparse.nn.functional.conv.conv_config import (
            Dataflow, set_global_conv_config, _default_conv_config
        )
        from torchsparse.nn.functional.conv.conv_mode import set_conv_mode
        hip_config = _default_conv_config.copy()
        hip_config['dataflow'] = Dataflow.GatherScatter
        set_global_conv_config(hip_config)
        set_conv_mode(0)
        print("[TORCHSPARSE] Configured for HIP: GatherScatter dataflow, mode0")
    except Exception as e:
        print(f"[TORCHSPARSE] Warning: Could not configure for HIP: {e}")

import imageio
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipeline.cuda()

# Run the pipeline
outputs = pipeline.run(
    "A chair looking like a avocado.",
    seed=1,
    # Optional parameters
    # sparse_structure_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

# Render the outputs
video = render_utils.render_video(outputs['gaussian'][0])['color']
imageio.mimsave("sample_gs.mp4", video, fps=30)
video = render_utils.render_video(outputs['radiance_field'][0])['color']
imageio.mimsave("sample_rf.mp4", video, fps=30)
video = render_utils.render_video(outputs['mesh'][0])['normal']
imageio.mimsave("sample_mesh.mp4", video, fps=30)

# GLB files can be extracted from the outputs
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    # Optional parameters
    simplify=0.95,          # Ratio of triangles to remove in the simplification process
    texture_size=1024,      # Size of the texture used for the GLB
)
glb.export("sample.glb")

# Save Gaussians as PLY files
outputs['gaussian'][0].save_ply("sample.ply")
