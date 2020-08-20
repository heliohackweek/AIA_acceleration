# adapted from register and psf examples in the AIApy Gallery
# at https://aiapy.readthedocs.io/en/latest/generated/gallery/index.html

# Serial pipeline example for registering lvl 1 AIA data from local storage
# (not optimized)

# lvl 1.5 files saved with same file name, except "lev1" -> "lev1.5"
# WARNING: lvl 1.5 files are 10x than lvl1 files (since they are in float64)
#  (there may be better way to save the lvl1.5 files?)


import astropy.units as u
from aiapy.calibrate import register, update_pointing
import re
import os
import sunpy.map
import aiapy.psf

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
def gen_lvl_1p5(lvl1_fname_list,lvl1_path):
    for af in lvl1_fname_list:
        m = sunpy.map.Map(lvl1_path+'/'+af)
        m_up = update_pointing(m)
        m_reg = register(m_up)
        m_norm = sunpy.map.Map(m_reg.data/m_reg.exposure_time.to(u.s).value,
                               m_reg.meta)
        new_name = af.replace('lev1', 'lev1.5')
        m_norm.save(lvl1_path+'/'+new_name)
        print('to lvl 1.5: {} at {}'.format(str(m_norm.wavelength),str(m_norm.date)))
    return

# deconvolution pipeline example; only does one wavelength at a time
#  (since deconvolution psf is unique for each wavelength)
# returns list of skipped filenames
def gen_deconv(lvl1_fname_list, lvl1_path):
    # do first image/calculate psf
    af = lvl1_fname_list[0]
    m = sunpy.map.Map(lvl1_path+"/"+af)
    wave = m.wavelength

    #slow without GPU
    psf = aiapy.psf.psf(wave)

    # finish for first image
    m_decov = aiapy.psf.deconvolve(m,psf=psf)
    new_name = af.replace('lev1', 'levDecv')
    m_decov.save(lvl1.path+'/'+new_name)
    print('deconvolved: {} at {}'.format(str(m_decov.wavelength),str(m_decov.date)))

    skipped_files = []
    # do rest of lvl1 stuff
    for af in lvl1_fname_list[1:]:
        # prob better way to check wavelength (via filename?)
        m = sunpy.map.Map(lvl1_path+"/"+af)
        if(m.wavelength != wave):
            skipped_files.append(af)
            continue
        m_decov = aiapy.psf.deconvolve(m,psf=psf)
        new_name = af.replace('lev1', 'levDecv')
        m_decov.save(lvl1.path+'/'+new_name)
        print('deconvolved: {} at {}'.format(str(m_decov.wavelength),str(m_decov.date)))
    
    return skipped_files
        
if __name__ == '__main__':

    lvl1_path = 'AIA_data/171A/'
    aia_files = scrape_AIA_files(lvl1_path)

    #for example, only do first five files
    aia_files_samp = aia_files[:5]

    #outputs: aia_lev1.5_*_lev1.5.fits 
    gen_lvl_1p5(aia_files,lvl1_path) 

    # WARNING: following takes many minutes to calculate psf
    #  and then a few minutes to deconvolve each image
    #  (expected run time: ~30-40 min if no GPU/multithreading)
    #  (a minute or two if GPU? <-- needs verification)
    #outputs: aia_levDecv_*_levDecv.fits
    """
    while aia_files_samp:
        aia_files_samp = gen_deconv(aia_files_samp, lvl1_path)
    """

