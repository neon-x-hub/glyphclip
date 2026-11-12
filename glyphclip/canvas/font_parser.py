"""Font metadata extraction, variable axis inspection, and weight sampling."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import io

from fontTools.ttLib import TTFont


@dataclass
class VariableAxis:
    """Represents a variable font axis (e.g. wght, wdth, slnt)."""
    tag: str
    min_value: float
    default_value: float
    max_value: float


@dataclass
class FontMetadata:
    """Metadata extracted from a TrueType or OpenType font."""
    family: str
    postscript_name: str
    subfamily: str = "Regular"
    weight: int = 400
    is_italic: bool = False
    is_variable: bool = False
    license: Optional[str] = None
    variable_axes: Dict[str, VariableAxis] = field(default_factory=dict)
    source_path: Optional[str] = None


class FontParser:
    """Parses font outlines and metadata using fontTools."""

    STANDARD_CSS_WEIGHTS: List[int] = [100, 200, 300, 400, 500, 600, 700, 800, 900]

    @staticmethod
    def _decode_name_record(record: Any) -> Optional[str]:
        """Decode a fontTools NameRecord safely into a string."""
        try:
            return record.toUnicode()
        except Exception:
            return None

    @classmethod
    def parse(cls, font_source: Union[str, Path, bytes, io.BytesIO]) -> FontMetadata:
        """Parse font metadata from a file path or in-memory byte buffer."""
        if isinstance(font_source, (str, Path)):
            font_path = str(font_source)
            tt = TTFont(font_path)
            source_path = font_path
        elif isinstance(font_source, bytes):
            tt = TTFont(io.BytesIO(font_source))
            source_path = None
        else:
            tt = TTFont(font_source)
            source_path = None

        family = "Unknown"
        subfamily = "Regular"
        postscript_name = "Unknown"
        license_text = None

        # Extract naming records
        if "name" in tt:
            names = tt["name"].names
            for record in names:
                val = cls._decode_name_record(record)
                if not val:
                    continue
                # 1: Font Family, 16: Preferred Family
                if record.nameID == 16:
                    family = val
                elif record.nameID == 1 and family == "Unknown":
                    family = val
                # 2: Subfamily, 17: Preferred Subfamily
                elif record.nameID == 17:
                    subfamily = val
                elif record.nameID == 2 and subfamily == "Regular":
                    subfamily = val
                # 6: PostScript Name
                elif record.nameID == 6 and postscript_name == "Unknown":
                    postscript_name = val
                # 13 or 14: License
                elif record.nameID in (13, 14) and license_text is None:
                    license_text = val

        if postscript_name == "Unknown":
            postscript_name = f"{family.replace(' ', '')}-{subfamily.replace(' ', '')}"

        # Extract weight from OS/2 table if available
        weight = 400
        is_italic = False
        if "OS/2" in tt:
            os2 = tt["OS/2"]
            if hasattr(os2, "usWeightClass") and os2.usWeightClass:
                weight = int(os2.usWeightClass)
            if hasattr(os2, "fsSelection"):
                # Bit 0 is Italic
                is_italic = bool(os2.fsSelection & 1)

        # Check for variable font axes (fvar table)
        is_variable = "fvar" in tt
        variable_axes: Dict[str, VariableAxis] = {}
        if is_variable:
            fvar = tt["fvar"]
            for axis in fvar.axes:
                variable_axes[axis.axisTag] = VariableAxis(
                    tag=axis.axisTag,
                    min_value=float(axis.minValue),
                    default_value=float(axis.defaultValue),
                    max_value=float(axis.maxValue),
                )

        return FontMetadata(
            family=family,
            postscript_name=postscript_name,
            subfamily=subfamily,
            weight=weight,
            is_italic=is_italic,
            is_variable=is_variable,
            license=license_text,
            variable_axes=variable_axes,
            source_path=source_path,
        )

    @classmethod
    def get_supported_weights(cls, metadata: FontMetadata) -> List[int]:
        """Return the list of standard CSS weights applicable for this font.
        
        For variable fonts, samples available standard weights within axis bounds.
        For static fonts, returns the single nominal weight.
        """
        if metadata.is_variable and "wght" in metadata.variable_axes:
            axis = metadata.variable_axes["wght"]
            return [
                w for w in cls.STANDARD_CSS_WEIGHTS
                if axis.min_value <= w <= axis.max_value
            ]
        return [metadata.weight]
