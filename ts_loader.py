"""
Load time series from Deutsche Bundesbank "compact" SDMX XML files.

The Bundesbank publishes series as XML with a fixed structure: one ``<bbk:Series>``
element holds metadata (title, frequency, unit) and many ``<bbk:Obs>`` rows with
``TIME_PERIOD`` and ``OBS_VALUE``. This module parses that into a tidy pandas
DataFrame plus a small ``SeriesMeta`` dataclass for display in the app.

Typical usage::

    df, meta = load_bbk_series(Path("data/euribor_3m_monthly_avg.xml"))
    # df has columns: date, value
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# XML namespace prefix "bbk" → full URI (required for ElementTree find/findall).
BBK_NS = {"bbk": "http://www.bundesbank.de/statistik/zeitreihen/BBKcompact"}


@dataclass(frozen=True)
class SeriesMeta:
    """
    Human-readable metadata for one Bundesbank series (from XML attributes).

    Attributes:
        bbk_id: Official series code, e.g. ``BBIG1.M.D0.EUR.MMKT.EURIBOR.M03.AVE.MA``.
        title_en: English title (falls back to German if English missing).
        title_de: German title string from the XML.
        freq: Letter code for frequency, e.g. ``M`` (monthly), ``D`` (daily).
        unit: Unit label, e.g. ``% p.a.``, ``USD``.
        unit_mult: Often a power-of-ten scale hint from the bank (string in XML).
    """

    bbk_id: str
    title_en: str
    title_de: str
    freq: str
    unit: str
    unit_mult: str


def data_dir() -> Path:
    """
    Return the path to the ``data/`` folder next to this file.

    The Streamlit app expects curated ``*.xml`` copies there.

    Returns:
        Absolute ``Path`` to the ``data`` directory (may or may not exist).
    """
    return Path(__file__).resolve().parent / "data"


def list_series_xml() -> list[Path]:
    """
    List all ``*.xml`` files in ``data_dir()``, sorted by name.

    Empty list if the directory is missing—caller should handle that.

    Returns:
        Sorted list of paths to XML files suitable for ``load_bbk_series``.
    """
    d = data_dir()
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob("*.xml") if p.is_file())


def load_bbk_series(path: Path | str) -> tuple[pd.DataFrame, SeriesMeta]:
    """
    Parse one Bundesbank compact XML file into a time-indexed series.

    Steps (high level):
        1. Parse XML and find the single ``bbk:Series`` element.
        2. Read metadata attributes into ``SeriesMeta``.
        3. Loop ``bbk:Obs`` children; keep rows with a numeric ``OBS_VALUE``.
        4. Build a DataFrame, sort by date ascending, drop duplicate dates (keep last).

    Args:
        path: Path to an ``.xml`` file (string or ``Path``).

    Returns:
        Tuple ``(df, meta)`` where ``df`` has columns ``date`` and ``value``.

    Raises:
        ValueError: If no series element exists or no numeric observations were found.
    """
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()

    # XPath with namespace: first Series anywhere under root.
    series_el = root.find(".//bbk:Series", BBK_NS)
    if series_el is None:
        raise ValueError(f"No bbk:Series found in {path}")

    # Attribute names use Bundesbank's BBK_* convention.
    meta = SeriesMeta(
        bbk_id=series_el.get("BBK_ID", ""),
        title_en=series_el.get("BBK_TITLE_ENG", "") or series_el.get("BBK_TITLE", ""),
        title_de=series_el.get("BBK_TITLE", ""),
        freq=series_el.get("FREQ", ""),
        unit=series_el.get("UNIT", ""),
        unit_mult=series_el.get("UNIT_MULT", ""),
    )

    rows: list[tuple[pd.Timestamp, float]] = []
    for obs in series_el.findall("bbk:Obs", BBK_NS):
        raw = obs.get("OBS_VALUE")
        if raw is None:
            # Weekends / missing prints sometimes omit the value attribute.
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        tp = obs.get("TIME_PERIOD")
        if not tp:
            continue
        rows.append((pd.to_datetime(tp), val))

    if not rows:
        raise ValueError(f"No numeric observations in {path}")

    df = pd.DataFrame(rows, columns=["date", "value"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    df = df.reset_index(drop=True)
    return df, meta
