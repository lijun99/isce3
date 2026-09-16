"""
Packing/interpretation of sub-swath validity within a combined InSAR mask.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


class SubswathMaskCodec(ABC):
    """
    Packs reference/secondary sub-swath numbers into, and decodes them
    back out of, one field of a combined InSAR mask.

    Different subclasses implement different packing schemes (e.g.
    decimal digits vs. bit-fields) behind the same interface, so
    generate_insar_mask() and interpret_subswath_mask()'s callers don't
    need to change when the packing scheme does.
    """

    @abstractmethod
    def pack(self, reference_num, secondary_num):
        """
        Pack sub-swath numbers into a field value.

        Parameters
        ----------
        reference_num : numpy.array of int
            Reference sub-swath number per pixel (0 means invalid).
        secondary_num : numpy.array of int
            Secondary sub-swath number per pixel, same shape.

        Returns
        -------
        numpy.ndarray
            uint32 field value, same shape as the inputs (not yet
            combined with any other packed fields, e.g. exception-mask
            bytes).
        """

    @abstractmethod
    def interpret(self, mask):
        """
        Decode sub-swath validity (and, if supported, water presence)
        from a packed mask.

        Parameters
        ----------
        mask : numpy.array
            A mask including both this codec's field and any other
            packed fields; only this codec's field is decoded.

        Returns
        -------
        reference_valid : bool
            True if the reference is valid, False otherwise.
        secondary_valid : bool
            True if the secondary is valid, False otherwise.
        water : bool or None
            True if water is present, False if absent, or None if this
            codec does not support a water flag.
        """


@dataclass(frozen=True)
class DigitSubswathMaskLayout(SubswathMaskCodec):
    """
    NISAR's current convention: sub-swath validity (and optionally a
    water flag) as decimal digits within one byte, produced by
    generate_insar_mask() in nisar/products/insar/utils.py.

    Each field's value must be a single decimal digit (0-9).
    """
    byte_mask: int = 0xFF
    secondary_place: int = 1
    reference_place: int = 10
    water_place: Optional[int] = 100
    nodata: int = 255

    def pack(self, reference_num, secondary_num):
        reference_num = np.asarray(reference_num)
        secondary_num = np.asarray(secondary_num)
        if np.any(reference_num >= 10) or np.any(secondary_num >= 10):
            raise ValueError(
                "DigitSubswathMaskLayout can only pack sub-swath numbers "
                "0-9 (one decimal digit each); a larger value would "
                "silently overflow into the adjacent digit")
        packed = (reference_num * self.reference_place +
                 secondary_num * self.secondary_place)
        return packed.astype(np.uint32) & np.uint32(self.byte_mask)

    def interpret(self, mask):
        nd = (mask == self.nodata)
        subswath_mask = np.asarray(mask & self.byte_mask)

        secondary_valid = (subswath_mask // self.secondary_place) % 10 != 0
        reference_valid = (subswath_mask // self.reference_place) % 10 != 0
        secondary_valid = np.where(nd, False, secondary_valid)
        reference_valid = np.where(nd, False, reference_valid)

        water = None
        if self.water_place is not None:
            water = (subswath_mask // self.water_place) % 10 != 0
            water = np.where(nd, False, water)

        return reference_valid, secondary_valid, water


NISAR_SUBSWATH_MASK_LAYOUT = DigitSubswathMaskLayout()


def pack_subswath_byte(reference_num, secondary_num,
                       layout=NISAR_SUBSWATH_MASK_LAYOUT):
    """Packs reference/secondary sub-swath numbers per `layout` (default:
    NISAR's convention). See SubswathMaskCodec.pack()."""
    return layout.pack(reference_num, secondary_num)


def interpret_subswath_mask(mask, layout=NISAR_SUBSWATH_MASK_LAYOUT):
    """Interprets a packed InSAR mask per `layout` (default: NISAR's
    convention). See SubswathMaskCodec.interpret()."""
    return layout.interpret(mask)
