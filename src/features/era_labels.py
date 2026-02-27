"""
NBA Era Labels
===============
Defines the six historical NBA eras and provides utilities to label any
dataset that contains a `season` column with the corresponding era metadata.

Era definitions are anchored to concrete rule changes and stylistic shifts:

  1. Foundational          1946–47 → 1953–54   (pre-shot clock)
  2. Shot Clock            1954–55 → 1978–79   (24-sec clock, pre-3PT)
  3. 3-Point Introduction  1979–80 → 1993–94   (3PT line added, Magic/Bird/Jordan)
  4. Physical / Isolation  1994–95 → 2003–04   (hand-checking, low pace)
  5. Open Court            2004–05 → 2014–15   (post hand-check rules, pace revival)
  6. 3-Point Revolution    2015–16 → present   (analytics-driven, record 3PA)

Season column format expected: 6-digit string like "202425" (YYYYYYYY) or
integer equivalent.  Both regular-season and playoff rows carry the same
season value so era labels apply uniformly.

Usage:
    from src.features.era_labels import label_eras, ERA_DEFINITIONS

    df = label_eras(df)          # adds era_num, era_name, era_start, era_end
    print(ERA_DEFINITIONS)       # inspect the full era table

    # or run standalone to preview the mapping:
    python src/features/era_labels.py
"""

import pandas as pd


# ── Era definitions ─────────────────────────────────────────────────────────────
# Each entry: (era_num, era_name, start_season_int, end_season_int, description)
# Season integers match the 6-digit format: e.g. 195455 for the 1954-55 season.

ERA_DEFINITIONS = pd.DataFrame([
    {
        "era_num":   1,
        "era_name":  "Foundational",
        "era_start": 194647,
        "era_end":   195354,
        "rule_anchor": "Pre-24-second shot clock",
        "description": (
            "The NBA's founding years. No shot clock meant teams could stall "
            "indefinitely, producing low-scoring, structurally incomparable games. "
            "Statistics from this era are largely non-comparable to later periods."
        ),
    },
    {
        "era_num":   2,
        "era_name":  "Shot Clock Era",
        "era_start": 195455,
        "era_end":   197879,
        "rule_anchor": "24-second shot clock introduced (1954-55)",
        "description": (
            "The 24-second shot clock transformed the game overnight — pace roughly "
            "doubled. Spans the Wilt Chamberlain and Bill Russell dynasties through "
            "to the Kareem Abdul-Jabbar era. The 3-point line does not yet exist."
        ),
    },
    {
        "era_num":   3,
        "era_name":  "3-Point Introduction",
        "era_start": 197980,
        "era_end":   199394,
        "rule_anchor": "3-point line adopted (1979-80)",
        "description": (
            "The ABA-imported 3-point line was added in 1979-80 but used sparingly. "
            "Covers the Magic vs. Bird rivalry, the Bad Boy Pistons, and Jordan's "
            "first three-peat. Still a mid-range and post-dominant era."
        ),
    },
    {
        "era_num":   4,
        "era_name":  "Physical / Isolation",
        "era_start": 199495,
        "era_end":   200304,
        "rule_anchor": "Hand-checking permitted; physical defense era",
        "description": (
            "Physical hand-checking defense was heavily permitted, slowing pace and "
            "suppressing scoring. ISO and post play dominated. Covers Jordan's second "
            "three-peat, the Shaq/Kobe Lakers dynasty, and peak Tim Duncan."
        ),
    },
    {
        "era_num":   5,
        "era_name":  "Open Court",
        "era_start": 200405,
        "era_end":   201415,
        "rule_anchor": "No-hand-check & defensive 3-second rules (2004-05)",
        "description": (
            "Rule changes in 2004-05 eliminated hand-checking and enforced the "
            "defensive 3-second violation, opening up the paint and rewarding "
            "athleticism. Pace and scoring rebounded. Positionless basketball and "
            "the spread offense began emerging. LeBron, Dirk, and early Steph."
        ),
    },
    {
        "era_num":   6,
        "era_name":  "3-Point Revolution",
        "era_start": 201516,
        "era_end":   999999,   # open-ended (current era)
        "rule_anchor": "Curry breaks 3PT record (2015-16); analytics-era onset",
        "description": (
            "Steph Curry's record-breaking 2015-16 season triggered a league-wide "
            "analytical shift. 3-point attempts per game hit all-time highs, pace "
            "accelerated, and small-ball lineups became standard. The game is faster, "
            "more spread, and more 3-point-dependent than at any prior point."
        ),
    },
])


# ── Season parsing ──────────────────────────────────────────────────────────────

def _to_season_int(season_val) -> int:
    """
    Normalize a season value to a 6-digit integer for era lookup.

    Accepts:
      - "202425"  →  202425   (already 6-digit string)
      - 202425    →  202425   (already int)
      - "2024-25" →  202425   (hyphenated — strips the dash)
      - "24-25"   →  202425   (short form — not typical in this project but handled)
    """
    s = str(season_val).replace("-", "").strip()
    if len(s) == 4:
        # e.g. "2425" → treat as "202425"
        s = "20" + s
    try:
        return int(s)
    except ValueError:
        return 0


# ── Main labeling function ──────────────────────────────────────────────────────

def label_eras(df: pd.DataFrame, season_col: str = "season") -> pd.DataFrame:
    """
    Add era metadata columns to any DataFrame that contains a season column.

    New columns added:
      era_num   (int 1–6)   — ordinal era index, useful for model encoding
      era_name  (str)       — human-readable era label
      era_start (int)       — era's first season as 6-digit int
      era_end   (int)       — era's last season as 6-digit int (999999 = current)

    Args:
        df:         DataFrame containing `season_col`
        season_col: Name of the column holding season values (default: "season")

    Returns:
        DataFrame with four new columns inserted after `season_col`.

    Raises:
        KeyError if `season_col` is not found in df.
    """
    if season_col not in df.columns:
        raise KeyError(
            f"Column '{season_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    season_ints = df[season_col].apply(_to_season_int)

    def _lookup(s_int):
        match = ERA_DEFINITIONS[
            (ERA_DEFINITIONS["era_start"] <= s_int) &
            (ERA_DEFINITIONS["era_end"]   >= s_int)
        ]
        if match.empty:
            return pd.Series({"era_num": 0, "era_name": "Unknown",
                               "era_start": 0, "era_end": 0})
        row = match.iloc[0]
        return pd.Series({
            "era_num":   row["era_num"],
            "era_name":  row["era_name"],
            "era_start": row["era_start"],
            "era_end":   row["era_end"],
        })

    era_cols = season_ints.apply(_lookup)

    # Insert era columns right after the season column
    insert_pos = df.columns.get_loc(season_col) + 1
    for i, col in enumerate(["era_num", "era_name", "era_start", "era_end"]):
        df.insert(insert_pos + i, col, era_cols[col])

    return df


def get_era(season_val) -> dict:
    """
    Look up the era for a single season value.

    Returns a dict with era_num, era_name, era_start, era_end, description.

    Example:
        get_era("202425")
        # → {'era_num': 6, 'era_name': '3-Point Revolution', ...}
    """
    s_int = _to_season_int(season_val)
    match = ERA_DEFINITIONS[
        (ERA_DEFINITIONS["era_start"] <= s_int) &
        (ERA_DEFINITIONS["era_end"]   >= s_int)
    ]
    if match.empty:
        return {"era_num": 0, "era_name": "Unknown", "description": "Season out of range"}
    return match.iloc[0].to_dict()


# ── Preview / entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("NBA Era Definitions")
    print("=" * 70)
    for _, row in ERA_DEFINITIONS.iterrows():
        end_label = "present" if row["era_end"] == 999999 else str(row["era_end"])
        print(f"\nEra {row['era_num']}: {row['era_name']}")
        print(f"  Seasons : {row['era_start']} → {end_label}")
        print(f"  Anchor  : {row['rule_anchor']}")
        print(f"  Summary : {row['description'][:100]}...")

    print("\n" + "=" * 70)
    print("Sample label_eras() output on test seasons:")
    test = pd.DataFrame({"season": [
        "194647", "195455", "197980", "199495", "200405", "201516", "202425"
    ]})
    test = label_eras(test)
    print(test[["season", "era_num", "era_name"]].to_string(index=False))
