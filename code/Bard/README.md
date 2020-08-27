# Codes
This is a bunch of programs for AIA stuff.

`basic_pipeline` is a simple (unoptimized) serial pipeline for converting level 1 AIA data to level 1.5 or level deconvolved. Modifications it needs:
1. save output files as np.int16 (not np.float64)
2. functions should to return processed maps instead of saving to disk (or have the option to)
3. Be aware that deconvolution needs to be done before register
4. rename functions to less confusing stuff

`update_register` contains several attempts to speed up the `aia.calibrate.register` function. Some notes:
1. `sunpy.map.maputils.contains_full_disk' needs to be rewritten, or at least have a faster version. I've used [Jack Ireland's implementation](https://github.com/wafels/sunpy/blob/faster_full_disk/sunpy/map/maputils.py#L149) (which was written during this hackweek!).
2. Need to fix the current openCV implementation to get similar results as the current register+`sunpy.map.rotate` flow.
3. The Cupy implementation seems strangely slow. Needs further profiling and benchmarking; I suspect that my rudimentary %timeit profiling is not being fair towards the GPU functions (awaiting R. Attie's more detailed look).

## Timing results
### single run
local machine, WSL+Ubuntu:
OG: ~12 sec
JI, new cfd: ~3sec
openCV: 1.5 sec 
(no cupy)

NVIDIA raplab:
OG: ~8 sec
JI, new cfd: ~2.1 sec
(no openCV)
cupy: either ~2.1 sec (first call) or 1.4 sec (subsequent)

cupy takes some time to compile the kernel, which is why it takes longer the first call.

### pipeline
on NVIDIA cluster (1 V100 GPU)
for 10 images in serial, total_time:
OG, order 3:  97s
OG, order 1:  91s
JI_cfd, order 3: 42s
JI_cfd, order 1: 36s
cupy, order 1: 32s


