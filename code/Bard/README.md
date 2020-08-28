# Codes
This is a bunch of programs for AIA stuff.

`basic_pipeline` is a simple (unoptimized) serial pipeline for converting level 1 AIA data to level 1.5 or level deconvolved. Modifications it needs:
1. save output files as np.int16 (not np.float64)
2. functions should to return processed maps instead of saving to disk (or have the option to)
3. Be aware that deconvolution needs to be done before register
4. rename functions to less confusing stuff
5. incorporate GPU flow (if available); otherwise do GPU deconv+CPU register

`update_register` contains several attempts to speed up the `aia.calibrate.register` function. Some notes:
1. `sunpy.map.maputils.contains_full_disk` needs to be rewritten, or at least have a faster version. I've used [Jack Ireland's implementation](https://github.com/wafels/sunpy/blob/faster_full_disk/sunpy/map/maputils.py#L149) (which was written during this hackweek!).
2. Need to fix the current openCV implementation to get similar `map.data` output as the current register+`sunpy.map.rotate` flow (there's a possible bug in the pixel centering/handling?)
3. The Cupy implementation seems strangely slow. Needs further profiling and benchmarking; I suspect that my rudimentary %timeit profiling is not being fair towards the GPU functions (awaiting R. Attie's more detailed look). Also, it is a very simple implementation and does not attempt optimization beyond copying data to GPU and running a `cupyx.scipy.ndimage.affine_transform`.

## Timing results
NOTE: I did not use the `cupy.prof` profiler; so these may be slightly unfair to the cupy version. [See R. Attie's more comprehensive timings](https://github.com/WaaallEEE/AIA_acceleration/blob/master/code/Bard/timeit_profiling.ipynb) [and other timings](https://github.com/WaaallEEE/AIA_acceleration/blob/master/code/Bard/cupy_register.ipynb).

### Packages
local machine:
* Python 3.6.9
* OpenCV-python 4.4.0

NVIDIA machine, singularity build:
* Python 3.5
* CuPy (stable version as of 2020/08/26)
* CUDA 11.1

### single image (test_update_register)
local machine, WSL+Ubuntu, relative timings:
|version|time|
|-----|-----|
|OG| ~4x|
|JI, new cfd| 1x|
|openCV + new cfd| 0.5x |
|(no cupy)|N/A|

NVIDIA raplab (v100):
|version|time|
|-----|-----|
|OG| ~4x |
|JI, new cfd| 1.0x|
|(no openCV)| N/A |
|cupy| either ~1.0x (first call) or 0.666x (subsequent)|

cupy takes some time to compile the kernel, which is why it takes longer the first call.

### basic pipeline (update_register_test_pipeline)
on NVIDIA cluster (1 V100 GPU)
for 10 images in serial, total_time:
|version|time|
|-----|-----|
|OG, order 3|  97s|
|OG, order 1|  91s|
|JI_cfd, order 3| 42s|
|JI_cfd, order 1| 36s|
|cupy, order 1| 32s|


