'''
Wrapper for dense offsets
'''

import pathlib
import time

import journal
import numpy as np
import isce3
import pycuampcor
from osgeo import gdal
from nisar.products.readers import SLC
from nisar.workflows.helpers import copy_raster, get_cfg_freq_pols
from nisar.workflows.yaml_argparse import YamlArgparse
from nisar.workflows.dense_offsets_runconfig import \
    DenseOffsetsRunConfig


def get_ampcor_slc(hdf5_file, freq, pol, lines_per_block, copy_path):
    '''
    Get the RSLC image for Ampcor: the HDF5 dataset, read directly by
    pycuampcor, or a memory mappable (ENVI) complex64 copy of a complex32
    dataset

    Parameters
    ----------
    hdf5_file: str
        Path to the RSLC HDF5 file
    freq: str
        Frequency band ('A' or 'B')
    pol: str
        Polarization
    lines_per_block: int
        Lines per block to copy the RSLC dataset
    copy_path: str
        Path to the memory mappable copy, if needed

    Returns
    -------
    str
        The image name for Ampcor: HDF5:<file>:<dataset>, or copy_path
    '''
    if not pycuampcor.has_hdf5:
        err_str = ('pycuampcor is built without HDF5 support and cannot read '
                   'the RSLC datasets; install pycuampcor with HDF5 '
                   '(e.g., from conda-forge)')
        journal.error('dense_offsets.get_ampcor_slc').log(err_str)
        raise RuntimeError(err_str)
    slc = SLC(hdf5file=hdf5_file)
    # pycuampcor reads float32 (complex64) datasets, not float16 (complex32)
    if not slc.is_dataset_complex32(freq, pol):
        return f'HDF5:{hdf5_file}:/{slc.slcPath(freq, pol)}'
    copy_raster(hdf5_file, freq, pol, lines_per_block, copy_path,
                file_type='ENVI')
    return copy_path


def run(cfg: dict):
    '''
    Run dense offsets
    '''

    # Pull parameters from cfg
    ref_hdf5 = cfg['input_file_group']['reference_rslc_file']
    sec_hdf5 = cfg['input_file_group']['secondary_rslc_file']
    scratch_path = pathlib.Path(cfg['product_path_group']['scratch_path'])
    offset_params = cfg['processing']['dense_offsets']

    # Initialize parameters shared between frequency A and B
    ref_slc = SLC(hdf5file=ref_hdf5)

    # Get coregistered SLC path
    coregistered_slc_path = pathlib.Path(offset_params['coregistered_slc_path'])

    error_channel = journal.error('dense_offsets.run')
    info_channel = journal.info('dense_offsets.run')
    info_channel.log('Start dense offsets estimation')

    # Check GPU use
    use_gpu = isce3.core.gpu_check.use_gpu(cfg['worker']['gpu_enabled'],
                                           cfg['worker']['gpu_id'])

    if use_gpu:
        # Set current CUDA device
        device = isce3.cuda.core.Device(cfg['worker']['gpu_id'])
        isce3.cuda.core.set_device(device)

    # Looping over frequencies and polarizations
    t_all = time.time()

    for freq, _, pol_list in get_cfg_freq_pols(cfg):
        offset_scratch = scratch_path / f'dense_offsets/freq{freq}'

        for pol in pol_list:
            # Set output directory and output filenames
            out_dir = offset_scratch / pol
            out_dir.mkdir(parents=True, exist_ok=True)

            # Reference SLC: the HDF5 dataset, or a memory mappable copy
            ref_raster_str = f'HDF5:{ref_hdf5}:/{ref_slc.slcPath(freq, pol)}'
            ref_raster = isce3.io.Raster(ref_raster_str)
            ref_raster_path = get_ampcor_slc(
                ref_hdf5, freq, pol, offset_params['lines_per_block'],
                str(out_dir / 'reference.slc'))

            # If running insar.py, a memory mappable second raster has been
            # created in the previous step (resample slc). If secondary raster
            # is extracted from HDF5 file, read the HDF5 dataset directly or
            # make a memory mappable copy
            if coregistered_slc_path.is_file():
                sec_raster_path = get_ampcor_slc(
                    sec_hdf5, freq, pol, offset_params['lines_per_block'],
                    str(out_dir / 'secondary.slc'))
            else:
                 sec_raster_path = str(coregistered_slc_path /
                                       f'coarse_resample_slc/freq{freq}/{pol}/coregistered_secondary.slc')
            sec_raster = isce3.io.Raster(sec_raster_path)

            # Create the Ampcor object from the images, the output
            # filenames (flat binaries, e.g. ENVI files) and the dense
            # offsets parameters
            ampcor = create_ampcor(
                use_gpu, cfg['worker']['gpu_id'],
                referenceImageName=ref_raster_path,
                referenceImageHeight=ref_raster.length,
                referenceImageWidth=ref_raster.width,
                secondaryImageName=sec_raster_path,
                secondaryImageHeight=sec_raster.length,
                secondaryImageWidth=sec_raster.width,
                offsetImageName=str(out_dir / 'dense_offsets'),
                grossOffsetImageName=str(out_dir / 'gross_offset'),
                snrImageName=str(out_dir / 'snr'),
                covImageName=str(out_dir / 'covariance'),
                peakValueImageName=str(out_dir / 'correlation_peak'),
                **get_ampcor_params(offset_params, ref_raster.length,
                                    ref_raster.width))

            # Create empty ENVI datasets. PyCuAmpcor will overwrite the
            # binary files. Note, use gdal to pass interleave option
            create_empty_dataset(str(out_dir / 'dense_offsets'),
                                 ampcor.numberWindowAcross,
                                 ampcor.numberWindowDown, 2, gdal.GDT_Float32)
            create_empty_dataset(str(out_dir / 'gross_offsets'),
                                 ampcor.numberWindowAcross,
                                 ampcor.numberWindowDown, 2, gdal.GDT_Float32)
            create_empty_dataset(str(out_dir / 'snr'),
                                 ampcor.numberWindowAcross,
                                 ampcor.numberWindowDown, 1, gdal.GDT_Float32)
            create_empty_dataset(str(out_dir / 'covariance'),
                                 ampcor.numberWindowAcross,
                                 ampcor.numberWindowDown, 3, gdal.GDT_Float32)
            create_empty_dataset(str(out_dir / 'correlation_peak'),
                                 ampcor.numberWindowAcross,
                                 ampcor.numberWindowDown, 1, gdal.GDT_Float32)
            # Run dense offsets
            ampcor.runAmpcor()

    t_all_elapsed = time.time() - t_all
    info_channel.log(
        f"Successfully ran dense_offsets in {t_all_elapsed:.3f} seconds")


def create_ampcor(use_gpu, gpu_id, **params):
    '''
    Create an Ampcor object set up to run

    Parameters
    ----------
    use_gpu: bool
        Whether to use the CUDA (PyCuAmpcor) or the CPU (PyCPUAmpcor)
        implementation
    gpu_id: int
        GPU device ID, if using the CUDA implementation
    params: dict
        Ampcor parameters (keyword arguments named as the properties of
        the Ampcor objects), with the gross offsets grossOffsetDown and
        grossOffsetAcross (constant, or one per window)

    Returns
    -------
    pycuampcor.PyCuAmpcor or pycuampcor.PyCPUAmpcor
        Ampcor object with all parameters set, ready to run
    '''
    if use_gpu:
        ampcor = pycuampcor.PyCuAmpcor(deviceID=gpu_id, **params)
    else:
        ampcor = pycuampcor.PyCPUAmpcor(**params)

    # Check pixel in image range; warns to stderr when out of range but does not abort
    ampcor.checkPixelInImageRange()
    return ampcor


def get_ampcor_params(cfg, length, width):
    '''
    Get the Ampcor parameters from the dense offsets cfg values

    Parameters
    ----------
    cfg: dict
        Dictionary with user-defined dense offsets parameters
    length: int
        Number of lines of the reference RSLC
    width: int
        Number of columns of the reference RSLC

    Returns
    -------
    dict
        Ampcor parameters, named as the properties of the Ampcor objects;
        the parameters not set in cfg keep the pycuampcor default values
    '''
    error_channel = journal.error('dense_offsets.get_ampcor_params')
    defaults = pycuampcor.PyCPUAmpcor()

    def value(key, default):
        return default if cfg[key] is None else cfg[key]

    # Window, search range and skip: the cfg values, or the defaults
    params = dict(
        windowSizeWidth=value('window_range', defaults.windowSizeWidth),
        windowSizeHeight=value('window_azimuth', defaults.windowSizeHeight),
        halfSearchRangeAcross=value('half_search_range',
                                    defaults.halfSearchRangeAcross),
        halfSearchRangeDown=value('half_search_azimuth',
                                  defaults.halfSearchRangeDown),
        skipSampleAcross=value('skip_range', defaults.skipSampleAcross),
        skipSampleDown=value('skip_azimuth', defaults.skipSampleDown))

    margin = value('margin', 0)

    # If gross offsets are set update margin
    if (cfg['gross_offset_range'] is not None) and (cfg['gross_offset_azimuth'] is not None):
        margin = max(margin, np.abs(cfg['gross_offset_range']),
                     np.abs(cfg['gross_offset_azimuth']))

    margin_rg = 2 * margin + 2 * params['halfSearchRangeAcross'] + params['windowSizeWidth']
    margin_az = 2 * margin + 2 * params['halfSearchRangeDown'] + params['windowSizeHeight']

    params['referenceStartPixelAcrossStatic'] = value(
        'start_pixel_range', margin + params['halfSearchRangeAcross'])
    params['referenceStartPixelDownStatic'] = value(
        'start_pixel_azimuth', margin + params['halfSearchRangeDown'])

    if cfg['offset_width'] is not None:
        params['numberWindowAcross'] = cfg['offset_width']
    else:
        offset_width = (width - margin_rg) // params['skipSampleAcross']
        if offset_width <= 0:
            err_str = (f"Image width ({width}) is too small for the configured parameters "
                       f"(margin_rg={margin_rg}): no valid ampcor windows fit across range.")
            error_channel.log(err_str)
            raise ValueError(err_str)
        params['numberWindowAcross'] = offset_width

    if cfg['offset_length'] is not None:
        params['numberWindowDown'] = cfg['offset_length']
    else:
        offset_length = (length - margin_az) // params['skipSampleDown']
        if offset_length <= 0:
            err_str = (f"Image length ({length}) is too small for the configured parameters "
                       f"(margin_az={margin_az}): no valid ampcor windows fit along azimuth.")
            error_channel.log(err_str)
            raise ValueError(err_str)
        params['numberWindowDown'] = offset_length

    params.update(get_ampcor_options(cfg))

    if cfg['slc_oversampling_factor'] is not None:
        params['rawDataOversamplingFactor'] = cfg['slc_oversampling_factor']

    if cfg['correlation_statistics_zoom'] is not None:
        params['corrStatWindowSize'] = cfg['correlation_statistics_zoom']

    if cfg['correlation_surface_zoom'] is not None:
        params['corrSurfaceZoomInWindow'] = cfg['correlation_surface_zoom']

    if cfg['correlation_surface_oversampling_factor'] is not None:
        params['corrSurfaceOverSamplingFactor'] = cfg[
            'correlation_surface_oversampling_factor']

    if cfg['windows_batch_range'] is not None:
        params['numberWindowAcrossInChunk'] = cfg['windows_batch_range']

    if cfg['windows_batch_azimuth'] is not None:
        params['numberWindowDownInChunk'] = cfg['windows_batch_azimuth']

    if cfg['cuda_streams'] is not None:
        params['nStreams'] = cfg['cuda_streams']

    # Gross offsets: constant, or one per window from a file
    if (cfg['use_gross_offsets'] is not None) and (
            cfg['gross_offset_range'] is not None) and \
            (cfg['gross_offset_azimuth'] is not None):
        params['grossOffsetDown'] = cfg['gross_offset_azimuth']
        params['grossOffsetAcross'] = cfg['gross_offset_range']

    if cfg['gross_offset_filepath'] is not None:
        params.update(read_gross_offsets(cfg['gross_offset_filepath'],
                                         params['numberWindowDown'],
                                         params['numberWindowAcross']))

    # If True, add constant slant range/azimuth gross offsets to
    # estimated dense offsets (will be used to resample slc)
    if cfg['merge_gross_offset'] is not None:
        params['mergeGrossOffset'] = bool(cfg['merge_gross_offset'])

    return params


def get_ampcor_options(cfg):
    '''
    Get the Ampcor options (cross-correlation domain and workflow, deramping
    method and axis, correlation surface oversampling method) set in cfg

    The options are given to pycuampcor by name: the cfg values, except for
    the cross-correlation domain 'spatial' (pycuampcor 'time') and the
    deramping method 'complex_no_deramp' (pycuampcor 'none')

    Parameters
    ----------
    cfg: dict
        Dictionary with user-defined dense offsets or offsets product
        parameters

    Returns
    -------
    dict
        Ampcor options, named as the properties of the Ampcor objects
    '''
    names = {'cross_correlation_domain': ('algorithm', {'spatial': 'time'}),
             'cross_correlation_workflow': ('workflow', {}),
             'deramping_method': ('derampMethod', {'complex_no_deramp': 'none'}),
             'deramping_axis': ('derampAxis', {}),
             'correlation_surface_oversampling_method':
                 ('corrSurfaceOverSamplingMethod', {})}
    options = {}
    for key, (name, renamed) in names.items():
        if cfg.get(key) is not None:
            options[name] = renamed.get(cfg[key], cfg[key])
    return options


def read_gross_offsets(filepath, offset_length, offset_width):
    '''
    Read the gross offsets, one (azimuth, range) pair of int32 per window

    Parameters
    ----------
    filepath: str
        Path to the binary file of the gross offsets
    offset_length: int
        Number of windows in azimuth
    offset_width: int
        Number of windows in range

    Returns
    -------
    dict
        The gross offsets grossOffsetDown and grossOffsetAcross, one per window
    '''
    gross_offset = np.fromfile(filepath, dtype=np.int32)
    windows_number = offset_length * offset_width
    if gross_offset.size != 2 * windows_number:
        err_str = "The input gross offset does not match the offset width*offset length"
        journal.error('dense_offsets.read_gross_offsets').log(err_str)
        raise RuntimeError(err_str)
    gross_offset = gross_offset.reshape(windows_number, 2)
    return dict(grossOffsetDown=gross_offset[:, 0],
                grossOffsetAcross=gross_offset[:, 1])


def create_empty_dataset(filename, width, length,
                         bands, dtype, interleave="bip", file_type="ENVI"):
    '''
    Create empty dataset with user-defined options
    '''
    driver = gdal.GetDriverByName(file_type)
    driver.Create(filename, xsize=width, ysize=length, bands=bands,
                  eType=dtype, options=[f"INTERLEAVE={interleave}"])


if __name__ == "__main__":
    '''
    Run dense offsets estimation
    '''
    # Load command line args
    dense_offsets_parser = YamlArgparse()
    args = dense_offsets_parser.parse()
    # Get cfg dict from CLI args
    dense_offsets_runconfig = DenseOffsetsRunConfig(args)
    # Run dense offsets
    run(dense_offsets_runconfig.cfg)
