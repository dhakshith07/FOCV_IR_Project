"""
workflow/zone_mapper.py
Maps SegFormer segmentation masks to named desk zones.
Status: STUB — Implementation in Phase 4.
"""


class ZoneMapper:
    """Converts SegFormer pixel masks to logical workspace zones."""

    STATUS = "NOT_STARTED"

    def map_zones(self, segmentation_mask) -> dict:
        """Return zone polygons from a segmentation mask."""
        raise NotImplementedError("ZoneMapper: Phase 4 not yet implemented.")
