# commands I used in iPython (prob should be a Juypter notebook?)
#caveats: openCV implementation has a wrong pixel center or padding (I think)
#         cupy implementation is not ideal, I just did the simplest 
#             possible array copying and function replacement

import update_register as ur
from aiapy.calibrate import register, update_pointing
import sunpy.map

path = './AIA_data/171A/aia_lev1_171a_2017_09_10t01_17_09_35z_image_lev1.fits'
m = sunpy.map.Map(path)
m_up = update_pointing(m)

%timeit m_r1 = register(m_up)
%timeit m_r2 = ur.cfd_register(m_up)
%timeit m_r3 = ur.CV_register(m_up)
%timeit m_r4 = ur.cupy_register(m_up)

"""
local machine, WSL:
OG: ~12 sec
JI, new cfd: ~3sec
openCV: 1.5 sec 
(no cupy)

NVIDIA raplab:
OG: ~8 sec
JI, new cfd: ~2.1 sec
(no openCV)
cupy: either ~2.1 sec (first call) or 1.4 sec (subsequent)

I think there's some caching going on for copying the array to GPU
(but there is definitely improvement possible for the cupy implementation)
"""
