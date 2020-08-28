# Accelerating Analysis of AIA Data Cubes with Cupy and Dask

Oftentimes, we want to look at AIA intensities as a function of time for many pixels across a region of interest,
e.g. an active region. This involves and aligning and stacking many images and then doing some operation along the
time axis of the stack. A few examples of this are calculating time lags in every pixel of an active region cutout or computing running differences across the entire solar disk.

## Four Possible scenarios to investigate

* Numpy
* Dask + Numpy
* Cupy
* Dask + Cupy

These three will obviously scale quite differently. I suspect the first and third will probably be the fastest with
smaller data cubes and the second and fourth will scale well for larger time intervals or regions of interest.

It would be nice to run this on a larger, multi-GPU setup, but those resources are limited at the moment.

These tests are being run on a Titan RTX with 24 GB of memory.

## Benchmarks
