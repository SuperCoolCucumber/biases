from __future__ import annotations

import pandas as pd


def assign_routing_split(
    frame: pd.DataFrame,
    *,
    calibration_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Assign the deterministic winner-stratified calibration/test split.

    This is shared by the full and pilot MT-Bench preparation paths so a row
    receives the same routing assignment regardless of which artifact is
    generated.
    """

    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1")
    if "winner" not in frame.columns:
        raise ValueError("frame must contain a winner column")

    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby("winner", dropna=False, group_keys=False):
        shuffled = group.sample(frac=1, random_state=seed).copy()
        calibration_size = round(len(shuffled) * calibration_fraction)
        calibration_size = (
            min(max(calibration_size, 1), len(shuffled) - 1)
            if len(shuffled) > 1
            else len(shuffled)
        )
        shuffled["routing_split"] = "test"
        shuffled.iloc[
            :calibration_size,
            shuffled.columns.get_loc("routing_split"),
        ] = "calibration"
        pieces.append(shuffled)

    if not pieces:
        return frame.assign(routing_split=pd.Series(dtype="object"))
    return (
        pd.concat(pieces, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
