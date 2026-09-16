import numpy as np
import pytest

from isce3.unwrap.subswath_mask import (
    DigitSubswathMaskLayout, NISAR_SUBSWATH_MASK_LAYOUT,
    interpret_subswath_mask, pack_subswath_byte,
)


def test_pack_interpret_round_trip():
    ref_num = np.array([0, 1, 2, 9, 3])
    sec_num = np.array([0, 1, 0, 5, 7])
    packed = pack_subswath_byte(ref_num, sec_num)
    ref_valid, sec_valid, water = interpret_subswath_mask(packed)
    assert np.array_equal(ref_valid, ref_num != 0)
    assert np.array_equal(sec_valid, sec_num != 0)
    assert np.array_equal(water, np.zeros_like(ref_num, dtype=bool))


def test_nodata_forces_invalid():
    ref_valid, sec_valid, water = interpret_subswath_mask(
        np.array([NISAR_SUBSWATH_MASK_LAYOUT.nodata]))
    assert ref_valid == [False]
    assert sec_valid == [False]
    assert water == [False]


def test_pack_out_of_range_raises():
    """Regression: a sub-swath number >= 10 used to silently overflow
    into the adjacent digit (e.g. reference_num=11 corrupted the water
    digit) instead of failing -- see pr-379-fix_insar_mask discussion."""
    with pytest.raises(ValueError):
        pack_subswath_byte(np.array([11]), np.array([2]))
    with pytest.raises(ValueError):
        pack_subswath_byte(np.array([2]), np.array([10]))


def test_custom_layout_round_trip():
    """A layout with different byte width, digit assignment, nodata, and
    water enabled round-trips correctly with no code changes -- the
    whole point of parameterizing by layout."""
    layout = DigitSubswathMaskLayout(
        byte_mask=0x0FFF, secondary_place=100, reference_place=1,
        water_place=10, nodata=4095)
    ref_num = np.array([0, 1, 2, 9, 3])
    sec_num = np.array([0, 1, 0, 5, 7])
    packed = pack_subswath_byte(ref_num, sec_num, layout=layout)
    ref_valid, sec_valid, water = interpret_subswath_mask(packed, layout=layout)
    assert np.array_equal(ref_valid, ref_num != 0)
    assert np.array_equal(sec_valid, sec_num != 0)
    assert np.array_equal(water, np.zeros_like(ref_num, dtype=bool))


def test_water_place_none_returns_none():
    layout = DigitSubswathMaskLayout(water_place=None)
    packed = pack_subswath_byte(np.array([1]), np.array([2]), layout=layout)
    _, _, water = interpret_subswath_mask(packed, layout=layout)
    assert water is None


def test_exception_mask_bits_do_not_corrupt_decode():
    """Regression: interpret_subswath_mask() used to decode decimal
    digits from the raw mask value without first isolating the
    sub-swath byte, so a nonzero exception-mask bit in bits 8-31
    corrupted reference_valid/secondary_valid. See
    isce3/unwrap/preprocess.py and nisar/products/utils.py history."""
    subswath_byte = pack_subswath_byte(np.array([1]), np.array([0]))  # ref valid, sec invalid
    full_mask = subswath_byte | np.uint32(1 << 16)  # unrelated reference exception bit set
    ref_valid, sec_valid, _ = interpret_subswath_mask(full_mask)
    assert ref_valid == [True]
    assert sec_valid == [False]
