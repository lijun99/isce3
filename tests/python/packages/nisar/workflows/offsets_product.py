#!/usr/bin/env python3

import pytest

from nisar.workflows.offsets_product import get_offsets_shape, get_start_pixels


def make_cfg(windows, start=None):
    '''Minimal offsets_product cfg with square layer windows'''
    cfg = {'margin': 50, 'gross_offset_range': 0, 'gross_offset_azimuth': 0,
           'start_pixel_range': start, 'start_pixel_azimuth': start,
           'skip_range': 15, 'skip_azimuth': 15}
    for i, win in enumerate(windows):
        cfg[f'layer{i + 1}'] = {'window_range': win, 'window_azimuth': win,
                                'half_search_range': 20,
                                'half_search_azimuth': 20}
    return cfg


@pytest.mark.parametrize("windows", [(32, 64, 128), (64, 96, 196),
                                     (33, 64, 127)])
@pytest.mark.parametrize("start", [None, 314])
def test_layer_windows_centered_on_grid(windows, start):
    cfg = make_cfg(windows, start)

    # Common grid center, as in helpers.get_offset_radar_grid
    az0, rg0 = get_start_pixels(cfg)
    center = rg0 + max(windows) // 2

    for win in windows:
        az_start, rg_start = get_start_pixels(cfg, win, win)
        assert rg_start + win // 2 == center
        assert az_start + win // 2 == center

    # Largest window keeps the common start pixel
    assert get_start_pixels(cfg, max(windows), max(windows)) == (az0, rg0)


@pytest.mark.parametrize("windows", [(32, 64, 128), (64, 96, 196)])
def test_layer_windows_inside_image(windows):
    cfg = make_cfg(windows)
    size = 2000
    off_length, off_width = get_offsets_shape(cfg, size, size)

    # Every layer window, first to last, lies within the image
    for win in windows:
        az_start, rg_start = get_start_pixels(cfg, win, win)
        assert min(az_start, rg_start) >= 0
        assert rg_start + (off_width - 1) * cfg['skip_range'] + win <= size
        assert az_start + (off_length - 1) * cfg['skip_azimuth'] + win <= size
