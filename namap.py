import matplotlib.pyplot as plt
from astropy import wcs 
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord

import numpy as np
import os
import copy

import src.detector as tod
import src.loaddata as ld
import src.mapmaker as mp
#import src.beam as bm
import src.pointing as pt 

import argparse
import warnings

SUPPRESS = '==SUPPRESS=='

OPTIONAL = '?'
ZERO_OR_MORE = '*'
ONE_OR_MORE = '+'
PARSER = 'A...'
REMAINDER = '...'

class CustomHelpFormatter(argparse.HelpFormatter):
    def _format_args(self, action, default_metavar):
        get_metavar = self._metavar_formatter(action, default_metavar)
        result = '%s' % get_metavar(1)
        if action.nargs is None:
            result = '%s' % get_metavar(1)
        elif action.nargs == OPTIONAL:
            result = '%s' % get_metavar(1)
        elif action.nargs == ZERO_OR_MORE:
            result = '%s [...]' % get_metavar(1)
        elif action.nargs == ONE_OR_MORE:
            result = '%s [...]' % get_metavar(1)
        elif action.nargs == REMAINDER:
            result = '...'
        elif action.nargs == PARSER:
            result = '%s ...' % get_metavar(1)
        elif action.nargs == SUPPRESS:
            result = ''
        else:
            try:
                formats = ['%s' for _ in range(action.nargs)]
            except TypeError:
                raise ValueError("invalid nargs value") from None
            result = ' '.join(formats) % get_metavar(action.nargs)
        
        return result

    def _format_action_invocation(self, action):
        if not action.option_strings:
            metavar, = self._metavar_formatter(action, action.dest)(1)
            return metavar

        else:
            parts = []

            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                parts.extend(action.option_strings)

            # if the Optional takes a value, format is:
            #    -s ARGS, --long ARGS
            else:
                #default = action.dest.upper()

                #args_string = _format_args(action, default)
                
                parts.append(action.option_strings[0].ljust(3))
                parts.append(action.option_strings[1])
                #parts[-1] += ' %s' % args_string
            return ', '.join(parts)

def make_wide(formatter, w=120, h=36):
    """Return a wider HelpFormatter, if possible."""
    try:
        kwargs = {'width': w, 'max_help_position': h}
        formatter(None, **kwargs)
        return lambda prog: formatter(prog, **kwargs)
    except TypeError:
        warnings.warn("argparse help formatter failed, falling back.")
        return formatter

def map2d(self, data=None, coord1=None, coord2=None, crval=None, ctype=None, pixnum=None, telcoord=False, cdelt=None, \
          crpix=None, projection=None, xystage=False, det_name=None, idx=None):

    '''
    Function to generate the map plots (I,Q and U) 
    when the plot button is pushed
    '''

    intervals = 3   

    if telcoord is False:        
        if xystage is False:
            position = SkyCoord(crval[0], crval[1], unit='deg', frame='icrs')

            size = (pixnum[1], pixnum[0])     # pixels

            cutout = Cutout2D(data, position, size, wcs=projection)
            proj = cutout.wcs
            mapdata = cutout.data
        else:
            masked = np.ma.array(data, mask=(np.abs(data)<1))
            mapdata = masked
            proj = 'rectilinear'

    else:
        idx_xmin = crval[0]-cdelt*pixnum[0]/2   
        idx_xmax = crval[0]+cdelt*pixnum[0]/2
        idx_ymin = crval[1]-cdelt*pixnum[1]/2
        idx_ymax = crval[1]+cdelt*pixnum[1]/2

        proj = None

        idx_xmin = np.amax(np.array([np.ceil(crpix[0]-1-pixnum[0]/2), 0.], dtype=int))
        idx_xmax = np.amin(np.array([np.ceil(crpix[0]-1+pixnum[0]/2), np.shape(data)[1]], dtype=int))

        if np.abs(idx_xmax-idx_xmin) != pixnum[0]:
            if idx_xmin != 0 and idx_xmax == np.shape(data)[1]:
                idx_xmin = np.amax(np.array([0., np.shape(data)[1]-pixnum[0]], dtype=int))
            if idx_xmin == 0 and idx_xmax != np.shape(data)[1]:
                idx_xmax = np.amin(np.array([pixnum[0], np.shape(data)[1]], dtype=int))

        idx_ymin = np.amax(np.array([np.ceil(crpix[1]-1-pixnum[1]/2), 0.], dtype=int))
        idx_ymax = np.amin(np.array([np.ceil(crpix[1]-1+pixnum[1]/2), np.shape(data)[0]], dtype=int))

        if np.abs(idx_ymax-idx_ymin) != pixnum[1]:
            if idx_ymin != 0 and idx_ymax == np.shape(data)[0]:
                idx_ymin = np.amax(np.array([0., np.shape(data)[0]-pixnum[1]], dtype=int))
            if idx_ymin == 0 and idx_ymax != np.shape(data)[0]:
                idx_ymax = np.amin(np.array([pixnum[1], np.shape(data)[0]], dtype=int))

        self.mapdata = data[idx_ymin:idx_ymax, idx_xmin:idx_xmax]
        crpix[0] -= idx_xmin
        crpix[1] -= idx_ymin

        w = wcs.WCS(naxis=2)
        w.wcs.crpix = crpix
        w.wcs.cdelt = cdelt
        w.wcs.crval = crval
        w.wcs.ctype = ["TLON-TAN", "TLAT-TAN"]
        proj = w

    levels = np.linspace(0.5, 1, intervals)*np.amax(self.mapdata)
    
    axis = plt.subplot(111, projection=proj)

    if telcoord is False:
        if ctype == 'RA and DEC':
            ra = axis.coords[0]
            dec = axis.coords[1]
            ra.set_axislabel('RA (deg)')
            dec.set_axislabel('Dec (deg)')
            dec.set_major_formatter('d.ddd')
            ra.set_major_formatter('d.ddd')
        
        elif ctype == 'AZ and EL':
            az = axis.coords[0]
            el = axis.coords[1]
            az.set_axislabel('AZ (deg)')
            el.set_axislabel('EL (deg)')
            az.set_major_formatter('d.ddd')
            el.set_major_formatter('d.ddd')
        
        elif ctype == 'CROSS-EL and EL':
            xel = axis.coords[0]
            el = axis.coords[1]
            xel.set_axislabel('xEL (deg)')
            el.set_axislabel('EL (deg)')
            xel.set_major_formatter('d.ddd')
            el.set_major_formatter('d.ddd')

        elif ctype == 'XY Stage':
            axis.set_title('XY Stage')
            axis.set_xlabel('X')
            axis.set_ylabel('Y')

    else:
        ra_tel = axis.coords[0]
        dec_tel = axis.coords[1]
        ra_tel.set_axislabel('YAW (deg)')
        dec_tel.set_axislabel('PITCH (deg)')
        ra_tel.set_major_formatter('d.ddd')
        dec_tel.set_major_formatter('d.ddd')

    if telcoord is False:
        im = axis.imshow(mapdata, origin='lower', cmap=plt.cm.viridis)
        axis.contour(mapdata, levels=levels, colors='white', alpha=0.5)
    else:
        im = axis.imshow(mapdata, origin='lower', cmap=plt.cm.viridis)
        axis.contour(mapdata, levels=levels, colors='white', alpha=0.5)
    plt.colorbar(im, ax=axis)
    title = 'Map '+ idx + ' det'+det_name     
    plt.title(title)

    path = os.getcwd()+'/plot/'+det_name+'_'+idx+'.png'

    plt.savefig(path)


parser = argparse.ArgumentParser(description='NaMap. A naive mapmaker written in Python for BLASTPol and BLAST-TNG', \
                                 formatter_class=make_wide(CustomHelpFormatter))

experiment_choice = ['BLASTPol', 'BLAST-TNG']
parser.add_argument('experiment', metavar = '', action='store', type=str, choices=experiment_choice, \
                    help='Choose the experiment between BLASTPol and BLAST-TNG')

### Data Paths Parameters
Repogroup = parser.add_argument_group('Data paths and selection')
Repogroup.add_argument('-te', '--telemetry', action='store_true', help='For BLAST-TNG, specify if the data are coming from \
                       telemetry, so mole, or if they are coming from hard drives')
Repogroup.add_argument('-dp', '--detector_path', metavar = '', action='store', type=str, help='Path for detector data')
Repogroup.add_argument('-cp', '--coordinates_path', metavar = '', action='store', type=str, help='Path for coordinates data')
Repogroup.add_argument('-d', '--detector', metavar = '', action='store', type=str, nargs='*', help='Detector name to be analyzed. \
                       Can be a single detector, a list of detector or and entire array. For this option just ask for ###_array, where \
                       ### is the wavelenght of the array to be analyzed. For telemetry data detector use a number between 1 to 5. \
                       For hard drive data use the real detector number, e.g. 0002 or 0964.')
Repogroup.add_argument('-r', '--roach_number', metavar = '', action='store', type=int, nargs='+', \
                       help='Roach number corrispondent to the detector(s) to be analyzed. \
                       For a full map from the 250um all the three roaches are required')
Repogroup.add_argument('-dt', '--detector_table', metavar = '', action='store', type=str, help='Path for the detector table')
Repogroup.add_argument('-pt', '--pointing_table', metavar = '', action='store', type=str, help='Path for the star camera offset table')

### Astronometry Parameter
coord_system = ['RA-DEC', 'AZ-EL', 'xEL-EL', 'XY']
Astrogroup = parser.add_argument_group('Astronometry parameters')
Astrogroup.add_argument('-c', '--coordinate_system', metavar='', action='store', choices=coord_system, type=str, \
                        help='Choice of coordinate system to draw a map')
Astrogroup.add_argument('-t', '--telescope_coordinate', action='store_true', \
                        help='Use of telescope coordinates to draw a map')
Astrogroup.add_argument('-g', '--gaussian_convolution', metavar='', action='store', choices=coord_system, type=float, \
                        help='STD of the gaussian used for the convolution. The value is in arcsec')
Astrogroup.add_argument('-rp', '--reference_pixel', metavar = '', action='store', type=float, nargs=4, \
                        help='Value of the reference pixel in degree and its associated pixel number. \
                        The input needs to be written as XX_deg YY_deg XX_pixel YY_pixel')
Astrogroup.add_argument('-ps', '--pixel_size', metavar='', action='store', type=float, nargs=2, \
                        help='Pixel size along the two axis in degree. The input needs to be written as XX YY')
Astrogroup.add_argument('-pn', '--pixel_number', metavar='', action='store', type=float, nargs=2, \
                        help='Number of pixels along the two axis. The input needs to be written as XX YY')
Astrogroup.add_argument('-i', '--i_only', action='store_true', \
                        help='If true only map in intensity is created')
Astrogroup.add_argument('-po', '--pointing_offset', action='store_true', \
                        help='If true a pointing offset for each detector is computed')

### Data parameters
Datagroup = parser.add_argument_group('Data parameters')
Datagroup.add_argument('-ca', '--coadd_detectors', action='store_false',\
                       help='Coadd detectors if multiple TODs are analyzed')
Datagroup.add_argument('-fd', '--frequency_detectors', metavar='', action='store', type=float,\
                       help='Frequency samples for detectors')
Datagroup.add_argument('-fa', '--frequency_acs', metavar='', action='store', type=float,\
                       help='Frequency samples for acs data')
Datagroup.add_argument('-fh', '--frequency_hwp', metavar='', action='store', type=float,\
                       help='Frequency samples for hwp')
Datagroup.add_argument('-fl', '--frequency_lst_lat', metavar='', action='store', type=float,\
                       help='Frequency samples for LST and LAT')
Datagroup.add_argument('-sd', '--sample_detectors', metavar='', action='store', type=float,\
                       help='Samples per frame in a dirfile for detectors')
Datagroup.add_argument('-sa', '--sample_acs', metavar='', action='store', type=float,\
                       help='Samples per frame in a dirfile for acs data')
Datagroup.add_argument('-sl', '--sample_lst_lat', metavar='', action='store', type=float,\
                       help='Samples per frame in a dirfile for LST and LAT')
Datagroup.add_argument('-sh', '--sample_hwp', metavar='', action='store', type=float,\
                       help='Samples per frame in a dirfile for hwp')
Datagroup.add_argument('-td', '--datatype_detectors', metavar='', action='store', type=str,\
                       help='Data type in a dirfile for detectors')
Datagroup.add_argument('-ls', '--datatype_lst', metavar='', action='store', type=str,\
                       help='Data type in a dirfile for LST')
Datagroup.add_argument('-la', '--datatype_lat', metavar='', action='store', type=str,\
                       help='Data type in a dirfile for LAT')
Datagroup.add_argument('-hw', '--datatype_hwp', metavar='', action='store', type=str,\
                       help='Data type in a dirfile for hwp')
Datagroup.add_argument('-ta', '--datatype_coord', metavar='', action='store', type=str, nargs=2, \
                       help='Data type in a dirfile for acs data. The input needs to be written as XX YY')
Datagroup.add_argument('-f', '--frames', metavar='', action='store', type=str, nargs=2, \
                       help='Starting frame and ending frame. The input needs to be written as XX YY')

### Preprocessing data analysis
TODgroup = parser.add_argument_group('TOD preproccesing parameters')
TODgroup.add_argument('-cf', '--cutoff_frequency', metavar='', action='store', type=float,\
                      help='Cutoff frequency for high-pass filter detector data')
TODgroup.add_argument('-pg', '--polynomial_order', metavar='', action='store', type=float, \
                      help='Polynomial order used to fit the TOD to detrend data')
TODgroup.add_argument('-to', '--time_offset', metavar='', action='store', type=float,\
                      help='Time delay between pointing solution and detector data')
TODgroup.add_argument('-ds', '--despike', metavar='', action='store', type=float, nargs='*', \
                      help='Despike flag. If called, two possible parameters can be explicited. \
                      The first is the height in sigma unit for the peak, and the second is the prominence \
                      in sigma unit for the peak')

args = parser.parse_args()
parameters = vars(args)

experiment = parameters['experiment']

if parameters['telemetry'] is not None:
    telemetry = True
else:
    telemetry = False

if parameters['detector'] not in ['250_array', '350_array', '500_array']:
    full_array=False
    det_list = parameters['detector'].copy()
else: 
    full_array=True

if parameters['detector_path'] is not None:
    detector_path = parameters['detector_path'].copy()

if parameters['coordinates_path'] is not None:
    coordinates_path = parameters['coordinates_path'].copy()

if parameters['roach_number'] is not None:
    roach_number = parameters['roach_number']
else:
    roach_number = None

if parameters['detector_table'] is not None:
    detector_table = parameters['detector_table'].copy()
else:
    detector_table = None

if parameters['pointing_table'] is not None:
    pointing_table = parameters['pointing_table'].copy()
else:
    pointing_table = None

if detector_table is not None or pointing_table is not None:
    correction = True
else:
    correction = False

if parameters['coordinate_system'] is not None:
    coordinate_system = parameters['coordinate_system'].copy()
    if coordinate_system == 'RA-DEC':
        coord1 = str('RA')
        coord2 = str('DEC')
        ctype = str('RA and DEC')
    elif coordinate_system == 'AZ-EL':
        coord1 = str('AZ')
        coord2 = str('EL')
        ctype = str('AZ and EL')
    elif coordinate_system == 'xEL-EL':
        coord1 = str('xEL')
        coord2 = str('EL')
        ctype = str('CROSS-EL and EL')
    elif coordinate_system == 'XY':
        coord1 = str('X')
        coord2 = str('Y')
        ctype = str('XY Stage')

if parameters['telescope_coordinate'] is not None:
    telescope_coordinate = parameters['telescope_coordinate']
else:
    telescope_coordinate=False

if parameters['gaussian_convolution'] is not None:
    std = parameters['gaussian_convolution']
    convolution = True
else:
    std = 0
    covolution = False

if parameters['reference_pixel'] is not None:
    reference_pixel = np.array(parameters['reference_pixel'])
    crval = reference_pixel[:2]
    crpix = reference_pixel[2:]

if parameters['pixel_size'] is not None:
    pixel_size = np.array(parameters['pixel_size'])

if parameters['pixel_number'] is not None:
    pixel_number = np.array(parameters['pixel_number'])

if parameters['i_only'] is not None:
    I_only = parameters['i_only']
else:
    I_only = False

if parameters['pointing_offset'] is not None:
    pointing_offset = parameters['pointing_offset']
else:
    pointing_offset = False

if parameters['coadd_detectors'] is not None:
    coadd_detectors = parameters['coadd_detectors']
else:
    coadd_detectors = False

if parameters['frequency_detector'] is not None:
    frequency_detector = parameters['frequency_detector']

if parameters['frequency_acs'] is not None:
    frequency_acs = parameters['frequency_acs']

if parameters['frequency_lst_lat'] is not None:
    frequency_lst_lat = parameters['frequency_lst_lat']
else:
    frequency_lst_lat = None

if parameters['frequency_hwp'] is not None:
    frequency_hwp = parameters['frequency_hwp']
else:
    frequency_hwp = 0

if parameters['sample_detector'] is not None:
    sample_detector = parameters['sample_detector']

if parameters['sample_acs'] is not None:
    sample_acs = parameters['sample_acs']

if parameters['sample_lst_lat'] is not None:
    sample_lst_lat = parameters['sample_lst_lat']
else:
    sample_lst_lat = None

if parameters['sample_hwp'] is not None:
    sample_hwp = parameters['sample_hwp']
else:
    sample_hwp = 0

if parameters['datatype_detectors'] is not None:
    datatype_detectors = parameters['datatype_detectors']

if parameters['datatype_lst'] is not None:
    datatype_lst = parameters['datatype_lst']
else:
    datatype_lst = None

if parameters['datatype_lat'] is not None:
    datatype_lat = parameters['datatype_lat']
else:
    datatype_lat = None

if parameters['datatype_hwp'] is not None:
    datatype_hwp = parameters['datatype_hwp']
else:
    datatype_hwp = None

if parameters['datatype_coord'] is not None:
    datatype_coord = parameters['datatype_coord']

if parameters['frames'] is not None:
    frames = parameters['frames']

if parameters['cutoff_frequency'] is not None:
    cutoff = parameters['cutoff_frequency']
else:
    cutoff = 0.1

if parameters['polynomial_order'] is not None:
    polynomial_order = parameters['polynomial_order']
else:
    polynomial_order = 5

if parameters['time_offset'] is not None:
    time_offset = parameters['time_offset']
else:
    time_offset = None

if parameters['despike'] is not None:
    despike = np.array(parameters['despike'])
    if despike[0] == 0 and despike[1] == 0:
        despike_bool = False
    else:
        despike_bool = True
else:
    despike[0] = 5
    despike[1] = 5
    despike_bool = True

if coadd_detectors is False:

    for i in range(len(det_list)):

        print('Load data, current progress', i/len(det_list))

        dataload = ld.data_value(detector_path, det_list[i], coordinates_path, \
                                 coord1, coord2, datatype_detectors, \
                                 datatype_coord[0], datatype_coord[1], \
                                 experiment, datatype_lst, datatype_lat, datatype_hwp, 
                                 frames[0], frames[1])

        if (correction and coord1.lower() == 'ra') or telescope_coordinate:
            det_data, coord1_data, coord2_data, hwp_data, lst_data, lat_data = dataload.values()

        else:
            det_data, coord1_data, coord2_data, hwp_data = dataload.values()
            lst_data = None
            lat_data = None

        if experiment.lower() == 'blast-tng':
            if coordinate_system == 'XY':
                xystage = True
            else:
                xystage = False
            zoomsyncdata = ld.frame_zoom_sync(det_data, frequency_detector, \
                                              sample_detector, coord1_data, \
                                              coord2_data, frequency_acs, 
                                              sample_acs, frames[0], \
                                              frames[1], experiment, \
                                              lst_data, lat_data, frequency_lst_lat, \
                                              sample_lst_lat, offset=time_offset, \
                                              roach_number = roach_number, \
                                              roach_pps_path = detector_path, xystage=xystage, \
                                              hwp_data=hwp_data, hwp_fs=frequency_hwp,\
                                              hwp_sample_frame=sample_hwp)
        elif experiment.lower() == 'blastpol':
            zoomsyncdata = ld.frame_zoom_sync(det_data, frequency_detector, \
                                              sample_detector, coord1_data, \
                                              coord2_data, frequency_acs, 
                                              sample_acs, frames[0], \
                                              frames[1], experiment, \
                                              lst_data, lat_data, frequency_lst_lat, \
                                              sample_lst_lat, offset=time_offset, \
                                              hwp_data=hwp_data, hwp_fs=frequency_hwp,\
                                              hwp_sample_frame=sample_hwp)

        if (correction and coord1.lower() == 'ra') or pointing_offset or telescope_coordinate:
            timemap, detslice, coord1slice, coord2slice, hwpslice, lstslice, latslice = zoomsyncdata.sync_data()
        else:
            timemap, detslice, coord1slice, coord2slice, hwpslice = zoomsyncdata.sync_data()
            lstslice = None
            latslice = None

        if datatype_hwp is not None:
            if experiment.lower() == 'blastpol':
                hwpslice = (hwpslice-0.451)*(-360.)

        if detector_table is not None:
            dettable = ld.det_table(det_list[i], experiment, detector_table)
            det_off, noise_det, grid_angle, pol_angle_offset, resp = dettable.loadtable()
        else:
            det_off = np.zeros((np.size(det_list[i]),2))
            noise_det = np.ones(np.size(det_list[i]))
            grid_angle = np.zeros(np.size(det_list[i]))
            pol_angle_offset = np.zeros(np.size(det_list[i]))
            resp = np.ones(np.size(det_list[i]))
                    
        if coord1.lower() == 'xel':
            coord1slice = coord1slice*np.cos(np.radians(coord2slice))

        if correction is True:
            if pointing_table is not None:               
                xsc_file = ld.xsc_offset(pointingoffsetnumber, frames[0], frames[1])
                xsc_offset = xsc_file.read_file()
            else:
                xsc_offset = np.zeros(2)

            corr = pt.apply_offset(coord1slice, coord2slice, datatype_coord, \
                                   xsc_offset, det_offset = det_off, lst = lstslice, \
                                   lat = latslice)

            coord1slice, coord2slice = corr.correction()
        else:
            if coord1.lower() == 'ra':
                coord1slice = coord1slice*15. #Conversion between hours to degree

        if telescope_coordinate or I_only is False:
            
            parallactic = np.zeros_like(coord1slice)
            if np.size(np.shape(detslice)) == 1:
                tel = pt.utils(coord1slice/15., coord2slice, lstslice, latslice)
                parallactic = tel.parallactic_angle()
            else:
                if np.size(np.shape(coord1slice)) == 1:
                    tel = pt.utils(coord1slice/15., coord2slice, lstslice, latslice)
                    parallactic = tel.parallactic_angle()
                else:
                    for j in range(np.size(np.shape(detslice))):
                        tel = pt.utils(coord1slice[j]/15., coord2slice[j], \
                                       lstslice, latslice)
                        parallactic[j,:] = tel.parallactic_angle()
        else:
            if np.size(np.shape(detslice)) == 1:
                parallactic = 0.
            else:
                if np.size(np.shape(coord1slice)) == 1:
                    parallactic = 0.
                else:
                    parallactic = np.zeros_like(detslice)


        det_tod = tod.data_cleaned(detslice, frequency_detector,cutoff, det_list[i],
                                   polynomial_order, despike_bool, despike[0], despike[1])
        cleaned_data = det_tod.data_clean()
        if np.size(resp) > 1:
            if experiment.currentText().lower() == 'blast-tng':
                cleaned_data = np.multiply(cleaned_data, np.reshape(1/resp, (np.size(1/resp), 1)))
            else:
                cleaned_data = np.multiply(cleaned_data, np.reshape(resp, (np.size(resp), 1)))
        else:
            if experiment.currentText().lower() == 'blast-tng':
                cleaned_data /= resp
            else:
                cleaned_data *= resp

        
        pol_angle = np.radians(parallactic+2*hwpslice+(grid_angle-2*pol_angle_offset))
        if np.size(np.shape(coord1slice)) != 1:
            pol_angle = np.reshape(pol_angle, np.size(pol_angle))

        maps = mp.maps(ctype, crpix, pixel_size, crval, cleaned_data, coord1slice, coord2slice, \
                       convolution, std, I_only, pol_angle=pol_angle, noise=noise_det, \
                       telcoord = telescope_coordinate, parang=parallactic)

        maps.wcs_proj()

        proj = maps.proj
        w = maps.w

        map_value = maps.map2d()

        x_min_map = np.floor(np.amin(w[:,0]))
        y_min_map = np.floor(np.amin(w[:,1]))
        index1, = np.where(w[:,0]<0)
        index2, = np.where(w[:,1]<0)

        if np.size(index1) > 1:
            crpix1_new  = (crpix[0]-x_min_map)
        else:
            crpix1_new = copy.copy(crpix[0])
        
        if np.size(index2) > 1:
            crpix2_new  = (crpix[1]-y_min_map)
        else:
            crpix2_new = copy.copy(crpix[1])

        crpix_new = np.array([crpix1_new, crpix2_new])

        if I_only:
            map2d(map_value, coord1=coord1slice, coord2=coord1slice, crval=crval, ctype=ctype, pixnum=pixel_number, \
                  telcoord=telescope_coordinate, crpix=crpix_new, cdelt=pixel_size, projection=proj, xystage=xystage, \
                  det_name=det_list[i], idx='I')
        else:
            idx_list = ['I', 'Q', 'U']

            for i in range(len(idx_list)):
                map2d(map_value, coord1=coord1slice, coord2=coord1slice, crval=crval, ctype=ctype, pixnum=pixel_number, \
                      telcoord=telescope_coordinate, crpix=crpix_new, cdelt=pixel_size, projection=proj, xystage=xystage, \
                      det_name=det_list[i], idx=idx_list[i])

else:
    dataload = ld.data_value(detector_path, det_list, coordinates_path, \
                             coord1, coord2, datatype_detectors, \
                             datatype_coord[0], datatype_coord[1], \
                             experiment, datatype_lst, datatype_lat, datatype_hwp, 
                             frames[0], frames[1], roach_number=roach_number, telemetry=telemetry)

    if (correction and coord1.lower() == 'ra') or telescope_coordinate:
        det_data, coord1_data, coord2_data, hwp_data, lst_data, lat_data = dataload.values()

    else:
        det_data, coord1_data, coord2_data, hwp_data = dataload.values()
        lst_data = None
        lat_data = None

    if experiment.lower() == 'blast-tng':
        if coordinate_system == 'XY':
            xystage = True
        else:
            xystage = False
        zoomsyncdata = ld.frame_zoom_sync(det_data, frequency_detector, \
                                          sample_detector, coord1_data, \
                                          coord2_data, frequency_acs, 
                                          sample_acs, frames[0], \
                                          frames[1], experiment, \
                                          lst_data, lat_data, frequency_lst_lat, \
                                          sample_lst_lat, offset=time_offset, \
                                          roach_number = roach_number, \
                                          roach_pps_path = detector_path, xystage=xystage, \
                                          hwp_data=hwp_data, hwp_fs=frequency_hwp,\
                                          hwp_sample_frame=sample_hwp)
    elif experiment.lower() == 'blastpol':
        zoomsyncdata = ld.frame_zoom_sync(det_data, frequency_detector, \
                                          sample_detector, coord1_data, \
                                          coord2_data, frequency_acs, 
                                          sample_acs, frames[0], \
                                          frames[1], experiment, \
                                          lst_data, lat_data, frequency_lst_lat, \
                                          sample_lst_lat, offset=time_offset, \
                                          hwp_data=hwp_data, hwp_fs=frequency_hwp,\
                                          hwp_sample_frame=sample_hwp)

    if (correction and coord1.lower() == 'ra') or pointing_offset or telescope_coordinate:
        timemap, detslice, coord1slice, coord2slice, hwpslice, lstslice, latslice = zoomsyncdata.sync_data(telemetry==telemetry)
    else:
        timemap, detslice, coord1slice, coord2slice, hwpslice = zoomsyncdata.sync_data(telemetry==telemetry)
        lstslice = None
        latslice = None

    if datatype_hwp is not None:
        if experiment.lower() == 'blastpol':
            hwpslice = (hwpslice-0.451)*(-360.)

    if detector_table is not None:
        dettable = ld.det_table(det_list, experiment, detector_table)
        det_off, noise_det, grid_angle, pol_angle_offset, resp = dettable.loadtable()
    else:
        det_off = np.zeros((np.size(det_list),2))
        noise_det = np.ones(np.size(det_list))
        grid_angle = np.zeros(np.size(det_list))
        pol_angle_offset = np.zeros(np.size(det_list))
        resp = np.ones(np.size(det_list))
                
    if coord1.lower() == 'xel':
        coord1slice = coord1slice*np.cos(np.radians(coord2slice))

    if correction is True:
        if pointing_table is not None:               
            xsc_file = ld.xsc_offset(pointingoffsetnumber, frames[0], frames[1])
            xsc_offset = xsc_file.read_file()
        else:
            xsc_offset = np.zeros(2)

        corr = pt.apply_offset(coord1slice, coord2slice, datatype_coord, \
                               xsc_offset, det_offset = det_off, lst = lstslice, \
                               lat = latslice)

        coord1slice, coord2slice = corr.correction()
    else:
        if coord1.lower() == 'ra':
            coord1slice = coord1slice*15. #Conversion between hours to degree

    if telescope_coordinate or I_only is False:
        
        parallactic = np.zeros_like(coord1slice)
        if np.size(np.shape(detslice)) == 1:
            tel = pt.utils(coord1slice/15., coord2slice, lstslice, latslice)
            parallactic = tel.parallactic_angle()
        else:
            if np.size(np.shape(coord1slice)) == 1:
                tel = pt.utils(coord1slice/15., coord2slice, lstslice, latslice)
                parallactic = tel.parallactic_angle()
            else:
                for j in range(np.size(np.shape(detslice))):
                    tel = pt.utils(coord1slice[j]/15., coord2slice[j], \
                                   lstslice, latslice)
                    parallactic[j,:] = tel.parallactic_angle()
    else:
        if np.size(np.shape(detslice)) == 1:
            parallactic = 0.
        else:
            if np.size(np.shape(coord1slice)) == 1:
                parallactic = 0.
            else:
                parallactic = np.zeros_like(detslice)

    det_tod = tod.data_cleaned(detslice, frequency_detector,cutoff, det_list,
                               polynomial_order, despike_bool, despike[0], despike[1])
    cleaned_data = det_tod.data_clean()
    if np.size(resp) > 1:
        if experiment.currentText().lower() == 'blast-tng':
            cleaned_data = np.multiply(cleaned_data, np.reshape(1/resp, (np.size(1/resp), 1)))
        else:
            cleaned_data = np.multiply(cleaned_data, np.reshape(resp, (np.size(resp), 1)))
    else:
        if experiment.currentText().lower() == 'blast-tng':
            cleaned_data /= resp
        else:
            cleaned_data *= resp

    if np.size(det_list) == 1:
        pol_angle = np.radians(parallactic+2*hwpslice+(grid_angle-2*pol_angle_offset))
        if np.size(np.shape(coord1slice)) != 1:
            pol_angle = np.reshape(pol_angle, np.size(pol_angle))
    else:
        pol_angle = np.zeros_like(cleaned_data)
        for i in range(np.size(det_list)):
            pol_angle[i,:] = np.radians(2*hwpslice+(grid_angle[i]-2*pol_angle_offset[i]))
            if np.size(np.shape(coord1slice)) == 1:
                pol_angle[i, :] += np.radians(parallactic)
            else:
                pol_angle[i, :] += np.radians(parallactic[i,:])

    maps = mp.maps(ctype, crpix, pixel_size, crval, cleaned_data, coord1slice, coord2slice, \
                   convolution, std, I_only, pol_angle=pol_angle, noise=noise_det, \
                   telcoord = telescope_coordinate, parang=parallactic)

    maps.wcs_proj()

    proj = maps.proj
    w = maps.w

    map_value = maps.map2d()

    x_min_map = np.floor(np.amin(w[:,0]))
    y_min_map = np.floor(np.amin(w[:,1]))
    index1, = np.where(w[:,0]<0)
    index2, = np.where(w[:,1]<0)

    if np.size(index1) > 1:
        crpix1_new  = (crpix[0]-x_min_map)
    else:
        crpix1_new = copy.copy(crpix[0])
    
    if np.size(index2) > 1:
        crpix2_new  = (crpix[1]-y_min_map)
    else:
        crpix2_new = copy.copy(crpix[1])

    crpix_new = np.array([crpix1_new, crpix2_new])

    if I_only:
        map2d(map_value, coord1=coord1slice, coord2=coord1slice, crval=crval, ctype=ctype, pixnum=pixel_number, \
              telcoord=telescope_coordinate, crpix=crpix_new, cdelt=pixel_size, projection=proj, xystage=xystage, \
              det_name=det_list[i], idx='I')
    else:
        idx_list = ['I', 'Q', 'U']

        for i in range(len(idx_list)):
            map2d(map_value, coord1=coord1slice, coord2=coord1slice, crval=crval, ctype=ctype, pixnum=pixel_number, \
                  telcoord=telescope_coordinate, crpix=crpix_new, cdelt=pixel_size, projection=proj, xystage=xystage, \
                  det_name=det_list[i], idx=idx_list[i])