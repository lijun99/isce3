import numpy as np
from scipy.ndimage import label as nd_label

from isce3.unwrap.bridge_phase import (bridgeConnectComponent,
                                       label_boundary,
                                       label_conn_comp)


def _two_blobs_and_a_sliver():
    """Two solid, well-separated 1000px blobs plus one thin (1px-wide,
    15px) sliver -- only the sliver should be dropped by any reasonable
    erosion; both blobs should always survive."""
    nrow, ncol = 60, 60
    unw = np.zeros((nrow, ncol), dtype=np.float64)
    unw[5:25, 5:55] = 1.0
    unw[35:55, 5:55] = 2.0
    unw[58, 5:20] = 3.0
    label_img, num_label = nd_label(unw != 0, structure=np.ones((3, 3)))
    assert num_label == 3
    return label_img, num_label


def test_label_conn_comp_erosion_keeps_all_surviving_regions():
    """Regression test: label_conn_comp's erosion-pruning step used to
    compare a *binary* eroded mask's trivial max (always 1) against
    original label ids, which silently dropped every region except
    whichever one scipy happened to number 1 -- regardless of whether
    other regions actually survived erosion. Two solid blobs plus one
    thin sliver should retain both blobs and drop only the sliver."""
    label_img, num_label = _two_blobs_and_a_sliver()

    result_img, result_num = label_conn_comp(
        label_img, min_num_pixel=5, erosion_size=3)

    assert result_num == 2
    sizes = sorted(np.bincount(result_img.ravel())[1:].tolist())
    assert sizes == [1000, 1000]


def test_label_boundary_erosion_keeps_all_surviving_regions():
    """Same regression as test_label_conn_comp_erosion_... but for
    label_boundary's (circular-SE) erosion-pruning stage, which had an
    identical bug."""
    label_img, num_label = _two_blobs_and_a_sliver()

    result_img, result_num, _ = label_boundary(
        label_img, num_label, erosion_size=3)

    assert result_num == 2
    sizes = sorted(np.bincount(result_img.ravel())[1:].tolist())
    assert sizes == [1000, 1000]


def test_bridge_connect_component_forwards_erosion_size():
    """Regression test: bridgeConnectComponent.label() used to not forward
    its erosion_size argument to label_conn_comp at all, so that stage
    always erodes with label_conn_comp's own default (5) regardless of
    what the caller passed -- e.g. erosion_size=0 (meaning "no erosion")
    did not actually disable the first-stage erosion."""
    label_img, num_label = _two_blobs_and_a_sliver()

    cc = bridgeConnectComponent(conncomp=label_img)
    cc.label(min_num_pixel=5, erosion_size=0)

    # with erosion fully disabled, the thin sliver survives on its own
    # (still above min_num_pixel=5, at 15px) alongside both blobs
    assert cc.num_label == 3


def test_label_conn_comp_square_se_is_radius_based_like_label_boundary():
    """Regression test: label_conn_comp's square structuring element used
    to be a literal erosion_size x erosion_size block (e.g. just 2x2 for
    erosion_size=2, non-centered) -- much gentler than label_boundary's
    radius-based, centered circular SE for the *same* erosion_size value
    (5x5 for erosion_size=2). A solid 3x3 (9px) region is too small to
    survive a proper centered 5x5 erosion (the SE itself doesn't fit
    inside it) but would trivially survive the old, much weaker literal
    2x2 SE -- directly distinguishing the two conventions by behavior."""
    nrow, ncol = 30, 30
    unw = np.zeros((nrow, ncol), dtype=np.float64)
    unw[10:13, 10:13] = 1.0   # solid 3x3 block, 9px

    label_img, num_label = nd_label(unw != 0, structure=np.ones((3, 3)))
    assert num_label == 1

    result_img, result_num = label_conn_comp(
        label_img, min_num_pixel=5, erosion_size=2)

    assert result_num == 0, (
        "a 3x3 region must NOT survive a proper radius-2 (5x5 centered) "
        "square erosion -- if it does, label_conn_comp's SE has regressed "
        "back to the old literal erosion_size x erosion_size convention"
    )
