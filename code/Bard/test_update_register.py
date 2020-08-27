# test update register functions; rudimentary timing

import numpy as np
import update_register as ur
from aiapy.calibrate import register, update_pointing
import sunpy.map
from timeit import default_timer as timer

if __name__=='__main__':
    
    path = './AIA_data/171A/aia_lev1_171a_2017_09_10t01_17_09_35z_image_lev1.fits'
    
    m = sunpy.map.Map(path)
    start = timer()
    m_up = update_pointing(m)
    end = timer()
    print("map time: {}".format(end-start))

    # NOTE: I also tested timings using %timeit <command> in iPython
    
    #OG
    start= timer()
    m_r1 = register(m_up)
    end= timer()
    print("OG time: {}".format(end-start))

    dat1 = m_r1.data[1000,1000]
    
    # with J.I.'s streamlined contains_full_disk
    start = timer()
    m_r2 = ur.cfd_register(m_up)
    end= timer()
    print("JI_cfd time: {}".format(end-start))
    dat2 = m_r2.data[1000,1000]
    
    # with J.I. cfd and R. Attie openCV imitation of map.rotate
    if ur.HAS_CV2:
        start = timer()
        m_r3 = ur.CV_register(m_up)
        end= timer()
        print("cv2 time: {}".format(end-start))
        dat3 = m_r3.data[1000,1000]
    else:
        dat3 = None

    #with cupy    
    if ur.HAS_CUPY:
        start = timer()
        m_r4 = ur.cupy_register(m_up, order=1) #only supports order 0 or 1
        end= timer()
        print("cupy time: {}".format(end-start))
        dat4 = m_r4.data[1000,1000]
    else:
        dat4 = None

    
    print(dat1,dat2,dat3,dat4)

    
