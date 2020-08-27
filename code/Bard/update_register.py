#WARNING: all edge cases untested.
#WARNING: still working on the CV_register function; currently does not get same results as the aiapy register function

# Bare bones register function with some improvements
# (courtesy of Jack Ireland, Raphael Attie, Chris Bard)

"""
code taken/modified from aiapy and sunpy
(locations are cited in the function header comments)
(under terms of BSD 2 Clause License;
   see licenses/sunpy.rst and licenses/aiapy.rst)
"""

import numpy as np
import astropy.units as u
from sunpy.map.sources.sdo import AIAMap, HMIMap
from sunpy.map import map_edges
import cv2

try:
    import cupy
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False    

"""
def OG_register:
just do `from aiapy.calibrate import register`
(https://gitlab.com/LMSAL_HUB/aia_hub/aiapy/-/blob/master/aiapy/calibrate/prep.py#L14)
"""

def cfd_register(smap, missing=None, order=3, use_scipy=False):
    """
    original: aiapy.calibrate.register
    ***MODIFIED FROM ORIGINAL; replaces contains_full_disk with new cfd***

    Processes a full-disk level 1 `~sunpy.map.sources.sdo.AIAMap` into a level
    1.5 `~sunpy.map.sources.sdo.AIAMap`.

    """
    # This implementation is taken directly from the `aiaprep` method in
    # sunpy.instr.aia.aiaprep under the terms of the BSD 2 Clause license.
    # See license in licenses/sunpy.rst
    if not isinstance(smap, (AIAMap, HMIMap)):
        raise ValueError("Input must be an AIAMap or HMIMap.")

    # EDIT: streamlined contains_full_disk
    # takes 903 msec (sunpy.map.contains_full_disk takes 12 seconds)
    if not JI_contains_full_disk(smap):
        raise ValueError("Input must be a full disk image.")

    # Target scale is 0.6 arcsec/pixel, but this needs to be adjusted if the
    # map has already been rescaled.
    if ((smap.scale[0] / 0.6).round() != 1.0 * u.arcsec / u.pix
            and smap.data.shape != (4096, 4096)):
        scale = (smap.scale[0] / 0.6).round() * 0.6 * u.arcsec
    else:
        scale = 0.6 * u.arcsec  # pragma: no cover # can't test this because it needs a full res image
    scale_factor = smap.scale[0] / scale

    missing = smap.min() if missing is None else missing

    tempmap = smap.rotate(recenter=True,
                          scale=scale_factor.value,
                          order=order,
                          missing=missing,
                          use_scipy=use_scipy)

    # extract center from padded smap.rotate output
    # crpix1 and crpix2 will be equal (recenter=True), as prep does not
    # work with submaps
    center = np.floor(tempmap.meta['crpix1'])
    range_side = (center + np.array([-1, 1]) * smap.data.shape[0] / 2) * u.pix
    newmap = tempmap.submap(
        u.Quantity([range_side[0], range_side[0]]),
        top_right=u.Quantity([range_side[1], range_side[1]]) - 1*u.pix)

    newmap.meta['r_sun'] = newmap.meta['rsun_obs'] / newmap.meta['cdelt1']
    newmap.meta['lvl_num'] = 1.5
    newmap.meta['bitpix'] = -64

    return newmap

def CV_register(smap, missing=None, order=3, use_scipy=False):
    """
    ***MODIFIED FROM ORIGINAL; streamlined contains_full_disk***
    *** and implements R. Attie's version using cv2 ***

    Processes a full-disk level 1 `~sunpy.map.sources.sdo.AIAMap` into a level
    1.5 `~sunpy.map.sources.sdo.AIAMap`.

    KEY DIFFERENCES TO smap.rotate:
    1. 
    2.

    """
    # This implementation is taken directly from the `aiaprep` method in
    # sunpy.instr.aia.aiaprep under the terms of the BSD 2 Clause license.
    # See license in file header
    if not isinstance(smap, (AIAMap, HMIMap)):
        raise ValueError("Input must be an AIAMap or HMIMap.")
    # EDIT: streamlined contains_full_disk
    if not JI_contains_full_disk(smap):
        raise ValueError("Input must be a full disk image.")

    # Target scale is 0.6 arcsec/pixel, but this needs to be adjusted if the
    # map has already been rescaled.
    if ((smap.scale[0] / 0.6).round() != 1.0 * u.arcsec / u.pix
            and smap.data.shape != (4096, 4096)):
        scale = (smap.scale[0] / 0.6).round() * 0.6 * u.arcsec
    else:
        scale = 0.6 * u.arcsec  # pragma: no cover # can't test this because it needs a full res image
    scale_factor = smap.scale[0] / scale

    missing = smap.min() if missing is None else missing

    # EDIT: changed from smap.rotate to scale_rotate

    """
    tempmap = smap.rotate(recenter=True,
                          scale=scale_factor.value,
                          order=order,
                          missing=missing,
                          use_scipy=use_scipy)
    """
    
    tempmap = scale_rotate(smap, scale_factor=scale_factor.value, missing=missing)
    
    # extract center from padded smap.rotate output
    # crpix1 and crpix2 will be equal (recenter=True), as prep does not
    # work with submaps
    center = np.floor(tempmap.meta['crpix1'])
    range_side = (center + np.array([-1, 1]) * smap.data.shape[0] / 2) * u.pix
    newmap = tempmap.submap(
        u.Quantity([range_side[0], range_side[0]]),
        top_right=u.Quantity([range_side[1], range_side[1]]) - 1*u.pix)

    newmap.meta['r_sun'] = newmap.meta['rsun_obs'] / newmap.meta['cdelt1']
    newmap.meta['lvl_num'] = 1.5
    newmap.meta['bitpix'] = -64

    return newmap

##-----------------------------------------------------
# Supporting functions

def JI_contains_full_disk(smap):
    """
    Copied from Jack Ireland's local implementation:
    https://github.com/wafels/sunpy/blob/faster_full_disk/sunpy/map/maputils.py#L127
    """
    # Calculate all the edge pixels
    top_, bottom, left_hand_side, right_hand_side = map_edges(smap)

    def _xy(ep):
        x = [p[0] for p in ep] * u.pix
        y = [p[1] for p in ep] * u.pix
        return x, y
    x, y = _xy(top_)
    horizontal1 = smap.pixel_to_world(x, y)
    
    x, y = _xy(bottom)
    horizontal2 = smap.pixel_to_world(x, y)
    
    x, y = _xy(left_hand_side)
    vertical1 = smap.pixel_to_world(x, y)
    
    x, y = _xy(right_hand_side)
    vertical2 = smap.pixel_to_world(x, y)
    
    radius = smap.rsun_obs
    
    # Determine the top and bottom edges of the map
    top = None
    bot = None
    if np.all(horizontal1.Ty > radius):
        top = horizontal1
    elif np.all(horizontal1.Ty < -radius):
        bot = horizontal1
        
    if np.all(horizontal2.Ty > radius):
        top = horizontal2
    elif np.all(horizontal2.Ty < -radius):
        bot = horizontal2
        
    # If either the top edge
    if top is None or bot is None:
        return False
    
    lhs = None
    rhs = None
    if np.all(vertical1.Tx > radius):
        rhs = vertical1
    elif np.all(vertical1.Tx < -radius):
        lhs = vertical1
    
    if np.all(vertical2.Tx > radius):
        rhs = vertical2
    elif np.all(vertical2.Tx < -radius):
        lhs = vertical2
        
    if lhs is None or rhs is None:
        return False
    
    return np.all(top.Ty > radius) and np.all(bot.Ty < -radius) and np.all(lhs.Tx < -radius) and np.all(rhs.Tx > radius)


def scale_rotate(smap, angle=None, scale_factor=1., missing=None):
    """
    Modified from R. Attie implementation
     At https://github.com/WaaallEEE/AIA-reloaded/blob/master/calibration.py
    and convolved with sunpy.map.mapbase.rotate

    DIFFERENCES between this and sunpy....rotate:
    1. Assumes recenter = True
    2. assumes order = 3 (cv.INTER_CUBIC)
    """
    if missing is None:
        missing = smap.min()

    if angle is None:
        ang = -smap.meta['CROTA2']
    elif angle is not None:
        ang = angle
        
    # convert angle to radian
    c = np.cos(np.deg2rad(ang))
    s = np.sin(np.deg2rad(ang))
    rmatrix = np.array([[c, -s],
                        [s, c]])
            
    array_center = (np.array(smap.data.shape)[::-1] - 1) / 2.0

    # The FITS-WCS transform is by definition defined around the
    # reference coordinate in the header.
    lon, lat = smap._get_lon_lat(smap.reference_coordinate.frame)
    rotation_center = u.Quantity([lon, lat])
    
    # Copy meta data
    new_meta = smap.meta.copy()
    
    extent = np.max(np.abs(np.vstack((smap.data.shape @ rmatrix,
                                      smap.data.shape @ rmatrix.T))), axis=0)

    # Calculate the needed padding or unpadding
    diff = np.asarray(np.ceil((extent - smap.data.shape) / 2), dtype=int).ravel()
    # Pad the image array
    pad_x = int(np.max((diff[1], 0)))
    pad_y = int(np.max((diff[0], 0)))

    new_meta['crpix1'] += pad_x
    new_meta['crpix2'] += pad_y
    
    new_data = np.pad(smap.data,
                      ((pad_y, pad_y), (pad_x, pad_x)),
                      mode='constant',
                      constant_values=(missing, missing))

    pixel_array_center = (np.flipud(new_data.shape) - 1) / 2.0
    
    # Create a temporary map so we can use it for the data to pixel calculation.
    temp_map = smap._new_instance(new_data, new_meta, smap.plot_settings)

    #this is same as `reference_pixel` in R. Attie original scale_rotate
    pixel_rotation_center = u.Quantity(temp_map.world_to_pixel(smap.reference_coordinate, origin=0)).value
    pixel_center = pixel_rotation_center
    
    del temp_map

    # DO CV THING HERE
    padded_array_center = (np.array(new_data.shape)[::-1] - 1) / 2.0
    padded_reference_pixel = pixel_rotation_center + np.array([pad_x, pad_y])
    rmatrix_cv = cv2.getRotationMatrix2D((padded_reference_pixel[0], padded_reference_pixel[1]), ang, scale_factor)

    # Adding extra shift to recenter:
    # move image so the reference pixel aligns with the center of the padded array
    shift = padded_array_center - padded_reference_pixel
    rmatrix_cv[0, 2] += shift[0]
    rmatrix_cv[1, 2] += shift[1]

    #cast new_data to float64, then warpAffine it
    new_data = new_data.astype(np.float64, casting='safe')
    
    rotated_image = cv2.warpAffine(new_data, rmatrix_cv, new_data.shape, cv2.INTER_CUBIC)
    
    new_reference_pixel = pixel_array_center
    
    new_meta['crval1'] = rotation_center[0].value
    new_meta['crval2'] = rotation_center[1].value
    new_meta['crpix1'] = new_reference_pixel[0] + 1  # FITS pixel origin is 1
    new_meta['crpix2'] = new_reference_pixel[1] + 1  # FITS pixel origin is 1
    
    # Unpad the array if necessary
    unpad_x = -np.min((diff[1], 0))
    if unpad_x > 0:
        new_data = new_data[:, unpad_x:-unpad_x]
        new_meta['crpix1'] -= unpad_x
    unpad_y = -np.min((diff[0], 0))
    if unpad_y > 0:
        new_data = new_data[unpad_y:-unpad_y, :]
        new_meta['crpix2'] -= unpad_y

    # Calculate the new rotation matrix to store in the header by
    # "subtracting" the rotation matrix used in the rotate from the old one
    # That being calculate the dot product of the old header data with the
    # inverse of the rotation matrix.
    pc_C = np.dot(smap.rotation_matrix, np.linalg.inv(rmatrix))
    new_meta['PC1_1'] = pc_C[0, 0]
    new_meta['PC1_2'] = pc_C[0, 1]
    new_meta['PC2_1'] = pc_C[1, 0]
    new_meta['PC2_2'] = pc_C[1, 1]

    # Update pixel size if image has been scaled.
    if scale_factor != 1.0:
        new_meta['cdelt1'] = (smap.scale[0] / scale_factor).value
        new_meta['cdelt2'] = (smap.scale[1] / scale_factor).value
    
    # Remove old CROTA kwargs because we have saved a new PCi_j matrix.
    new_meta.pop('CROTA1', None)
    new_meta.pop('CROTA2', None)
    # Remove CDi_j header
    new_meta.pop('CD1_1', None)
    new_meta.pop('CD1_2', None)
    new_meta.pop('CD2_1', None)
    new_meta.pop('CD2_2', None)
    
    # Create new map with the modification
    new_map = smap._new_instance(new_data, new_meta, smap.plot_settings)
    
    return new_map
    
# my testing says this takes about the same time as np.pad (so I've kept np.pad above)
def aia_pad(image, pad_x, pad_y, missing):
    newsize = [image.shape[0]+2*pad_y, image.shape[1]+2*pad_x]
    pimage = np.empty(newsize, dtype=image.dtype)
    pimage[0:pad_y,:] = missing
    pimage[:,0:pad_x]=missing
    pimage[pad_y+image.shape[0]:, :] = missing
    pimage[:, pad_x+image.shape[1]:] = missing
    pimage[pad_y:image.shape[0]+pad_y, pad_x:image.shape[1]+pad_x] = image
    return pimage
