import numpy as np
import update_register as ur
from aiapy.calibrate import register, update_pointing
import sunpy.map

if __name__=='__main__':
    path = './aia_lev1_171a_2017_09_10t01_16_09_35z_image_lev1.fits'
    m = sunpy.map.Map(path)
    m_up = update_pointing(m)

    # NOTE: I tested timings using %timeit <command> in iPython
    
    #OG
    m_r1 = register(m_up)
    # with J.I.'s streamlined contains_full_disk
    m_r2 = ur.cfd_register(m_up)
    # with J.I. cfd and R. Attie openCV imitation of map.rotate
    m_r3 = ur.CV_register(m_up)


