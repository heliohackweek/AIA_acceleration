#copied from basic_pipeline.py in same directory
# modified to test multiple image serial registration
# and rudimentarily time it

"""
on NVIDIA cluster (1 V100 GPU)
for 10 images in serial, total_time:
OG, order 3:  97s
OG, order 1:  91s
JI_cfd, order 3: 42s
JI_cfd, order 1: 36s
cupy, order 1: 32s

CONCLUSION: cupy (using straight replacement for scipy.nidmage.affine_transform) seems a bit slow? I need more detailed profiling (and perhaps better memory management)
Also, the new implementation of contains_full_disk (cfd) should be put in aiapy.
"""

import astropy.units as u
from aiapy.calibrate import register, update_pointing
from update_register import cfd_register, cupy_register
import re
import os
import sunpy.map
import aiapy.psf
from timeit import default_timer as timer

#cheap filename scraper
#assumes AIA data starts with 'aia_lev1_' (and that nothing else does)
def scrape_AIA_files(path):
    print(path)
    file_list = os.listdir(path)
    pattern = re.compile('aia_lev1_')
    aia_files = list(filter(pattern.match, file_list))

    if len(aia_files) == 0:
        raise LookupError("there were no aia data files")
    return aia_files


# converts lvl1 files in lvl1_path/lvl1_fname_list to lvl1.5
# puts lvl1.5 files in same directory as lvl1 (but named aia_lev1.5)
def gen_lvl_1p5(lvl1_fname_list,lvl1_path, lvl5_path=None):
    if lvl5_path is None:
        lvl5_path = lvl1_path
    for af in lvl1_fname_list:
        m = sunpy.map.Map(lvl1_path+'/'+af)
        m_up = update_pointing(m)
        m_reg = register(m_up, order=3)
        m_norm = sunpy.map.Map(m_reg.data/m_reg.exposure_time.to(u.s).value,
                               m_reg.meta)
        new_name = af.replace('lev1', 'lev1.5')
        m_norm.save(lvl5_path+'/'+new_name)
        print('to lvl 1.5: {} at {}'.format(str(m_norm.wavelength),str(m_norm.date)))
    return

#same as above, but with update_register.cfd_register
def gen_lvl_1p5_cfd(lvl1_fname_list,lvl1_path, lvl5_path=None):
    if lvl5_path is None:
        lvl5_path = lvl1_path
    for af in lvl1_fname_list:
        m = sunpy.map.Map(lvl1_path+'/'+af)
        m_up = update_pointing(m)
        m_reg = cfd_register(m_up, order=3)
        m_norm = sunpy.map.Map(m_reg.data/m_reg.exposure_time.to(u.s).value,
                               m_reg.meta)
        new_name = af.replace('lev1', 'lev1.5')
        m_norm.save(lvl5_path+'/'+new_name)
        print('to lvl 1.5: {} at {}'.format(str(m_norm.wavelength),str(m_norm.date)))
    return

#same as above, but with update_register.cupy_register
def gen_lvl_1p5_cupy(lvl1_fname_list,lvl1_path, lvl5_path=None):
    if lvl5_path is None:
        lvl5_path = lvl1_path
    for af in lvl1_fname_list:
        m = sunpy.map.Map(lvl1_path+'/'+af)
        m_up = update_pointing(m)
        m_reg = cupy_register(m_up, order=1)
        m_norm = sunpy.map.Map(m_reg.data/m_reg.exposure_time.to(u.s).value,
                               m_reg.meta)
        new_name = af.replace('lev1', 'lev1.5')
        m_norm.save(lvl5_path+'/'+new_name)
        print('to lvl 1.5: {} at {}'.format(str(m_norm.wavelength),str(m_norm.date)))
    return

if __name__ == '__main__':
    #input file directory
    lvl1_path = './AIA_data/'

    #output file directories
    lvl1p5_path = 'AIA_data/lvl1.5/'
    lvl1p5_path_cfd = 'AIA_data/lvl1.5_cfd/'
    lvl1p5_path_cupy = 'AIA_data/lvl1.5_cupy/'
    
    aia_files = scrape_AIA_files(lvl1_path)

    #for example, only do first five files
    aia_files_samp = aia_files[:5]

    #outputs: aia_lev1.5_*_lev1.5.fits
    start = timer()
    gen_lvl_1p5(aia_files_samp,lvl1_path, lvl1p5_path)
    end1 = timer()
    start2 = timer()
    gen_lvl_1p5_cfd(aia_files_samp,lvl1_path, lvl1p5_path_cfd)
    end2 = timer()
    start3 = timer()
    gen_lvl_1p5_cupy(aia_files_samp,lvl1_path, lvl1p5_path_cupy) 
    end3 = timer()
    
    print("end 1->1.5; OG: {}s; JIcfd: {}s, cupy: {}s".format(end1-start,end2-start2, end3-start3))
    
