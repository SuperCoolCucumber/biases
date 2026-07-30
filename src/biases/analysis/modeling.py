from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import math
import multiprocessing
import random
from typing import Any
import warnings


FIXED_EFFECTS_FORMULA = "flip ~ dose * family * congruence"
MIXED_EFFECTS_FORMULA = "flip ~ dose * family * congruence + (1 | question)"


class OptionalAnalysisDependencyError(ImportError):
    pass


@dataclass(frozen=True, slots=True)
class GEECoefficient:
    term: str
    estimate: float
    standard_error: float
    z_value: float
    p_value: float


@dataclass(frozen=True, slots=True)
class GEEResult:
    formula: str
    group_column: str
    n: int
    coefficients: tuple[GEECoefficient, ...]


@dataclass(frozen=True, slots=True)
class MixedLogitResult:
    formula: str
    group_column: str
    n: int
    coefficients: tuple[GEECoefficient, ...]
    fit_method: str
    converged: bool = True
    warnings: tuple[str, ...] = ()


UNCERTAINTY_GEE_FORMULA = "uncertainty ~ normalized_dose"


@dataclass(frozen=True, slots=True)
class UncertaintyGEEResult:
    formula: str
    group_column: str
    n: int
    n_clusters: int
    intercept: float
    slope: float
    slope_standard_error: float
    slope_z_value: float
    slope_p_value_one_sided: float
    converged: bool
    warnings: tuple[str, ...] = ()


ClusterIndexDraw = tuple[int, ...]
ClusteredUncertaintyRows = tuple[
    tuple[tuple[float, float], ...],
    ...,
]


def _cluster_index_draws(
    *,
    n_clusters: int,
    n_resamples: int,
    seed: int,
) -> tuple[ClusterIndexDraw, ...]:
    """Generate the complete serial bootstrap draw stream before dispatch."""

    rng = random.Random(seed)
    return tuple(
        tuple(rng.randrange(n_clusters) for _ in range(n_clusters))
        for _ in range(n_resamples)
    )


def _contiguous_draw_chunks(
    draws: Sequence[ClusterIndexDraw],
    *,
    workers: int,
) -> tuple[tuple[ClusterIndexDraw, ...], ...]:
    """Split ordered draws without changing their eventual result order."""

    chunk_count = min(workers, len(draws))
    base_size, remainder = divmod(len(draws), chunk_count)
    chunks: list[tuple[ClusterIndexDraw, ...]] = []
    start = 0
    for index in range(chunk_count):
        size = base_size + (1 if index < remainder else 0)
        stop = start + size
        chunks.append(tuple(draws[start:stop]))
        start = stop
    return tuple(chunks)


def _fit_uncertainty_gee_draw_chunk(
    clustered_rows: ClusteredUncertaintyRows,
    draws: Sequence[ClusterIndexDraw],
) -> tuple[float | None, ...]:
    """Fit one contiguous chunk of pre-generated cluster draws."""

    slopes: list[float | None] = []
    for draw in draws:
        sampled_rows: list[dict[str, Any]] = []
        for draw_index, cluster_index in enumerate(draw):
            sampled_rows.extend(
                {
                    "question_id": f"bootstrap-{draw_index}",
                    "normalized_dose": normalized_dose,
                    "uncertainty": uncertainty,
                }
                for normalized_dose, uncertainty in clustered_rows[cluster_index]
            )
        try:
            result = fit_uncertainty_gee(sampled_rows)
        except (
            OptionalAnalysisDependencyError,
            ValueError,
            RuntimeError,
        ):
            slopes.append(None)
            continue
        slopes.append(result.slope if math.isfinite(result.slope) else None)
    return tuple(slopes)


def cluster_bootstrap_uncertainty_gee_slopes(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_resamples: int,
    seed: int,
    workers: int = 1,
    group_column: str = "question_id",
) -> tuple[float | None, ...]:
    """Fit GEE slopes for deterministic question-cluster bootstrap draws.

    Draws are generated entirely in the calling process, then divided into
    contiguous chunks. Collecting those chunks in submission order makes the
    returned sequence independent of worker scheduling and byte-identical to
    the serial result.
    """

    if workers < 1:
        raise ValueError("workers must be a positive integer")
    if n_resamples < 1:
        return ()
    grouped: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        missing = {
            group_column,
            "normalized_dose",
            "uncertainty",
        } - set(row)
        if missing:
            raise ValueError(
                "uncertainty bootstrap input is missing columns: "
                f"{sorted(missing)}"
            )
        grouped.setdefault(row[group_column], []).append(row)
    keys = sorted(grouped, key=repr)
    if len(keys) < 2:
        return ()
    clustered_rows: ClusteredUncertaintyRows = tuple(
        tuple(
            (
                float(row["normalized_dose"]),
                float(row["uncertainty"]),
            )
            for row in grouped[key]
        )
        for key in keys
    )
    draws = _cluster_index_draws(
        n_clusters=len(keys),
        n_resamples=n_resamples,
        seed=seed,
    )
    chunks = _contiguous_draw_chunks(draws, workers=workers)
    if len(chunks) == 1:
        return _fit_uncertainty_gee_draw_chunk(clustered_rows, chunks[0])
    with ProcessPoolExecutor(
        max_workers=len(chunks),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = tuple(
            executor.submit(
                _fit_uncertainty_gee_draw_chunk,
                clustered_rows,
                chunk,
            )
            for chunk in chunks
        )
        return tuple(
            slope
            for future in futures
            for slope in future.result()
        )


def _modeling_frame(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_column: str,
) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise OptionalAnalysisDependencyError(
            "mixed-effects analysis requires optional dependency 'pandas'"
        ) from exc
    frame = pd.DataFrame([dict(row) for row in rows])
    required = {"flip", "dose", "family", "congruence", group_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"modeling input is missing required columns: {missing}")
    frame = frame.dropna(subset=list(required)).copy()
    if frame.empty:
        raise ValueError("modeling input has no complete rows")
    frame["flip"] = frame["flip"].astype(int)
    return frame


def fit_flip_mixed_logit(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_column: str = "question_id",
) -> MixedLogitResult:
    """Fit the predeclared random-intercept binomial mixed model.

    Statsmodels expresses the fixed and random components separately. The
    reported paper formula is ``flip ~ dose * family * congruence +
    (1 | question)``; ``0 + C(question_id)`` implements that question-level
    random intercept in ``BinomialBayesMixedGLM``.
    """

    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise OptionalAnalysisDependencyError(
            "fit_flip_mixed_logit requires optional dependency 'statsmodels'"
        ) from exc
    frame = _modeling_frame(rows, group_column=group_column)
    model = BinomialBayesMixedGLM.from_formula(
        FIXED_EFFECTS_FORMULA,
        {"question": f"0 + C({group_column})"},
        frame,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fitted = model.fit_vb()
    warning_messages = tuple(
        dict.fromkeys(str(item.message) for item in captured)
    )
    optim_retvals = getattr(fitted, "optim_retvals", None)
    converged = bool(
        optim_retvals.get("success", True)
        if isinstance(optim_retvals, Mapping)
        else True
    )
    if not converged:
        detail = "; ".join(warning_messages[:3])
        raise RuntimeError(
            "mixed-effects logistic fit did not converge"
            + (f": {detail}" if detail else "")
        )
    coefficients: list[GEECoefficient] = []
    for term, estimate, standard_error in zip(
        model.exog_names,
        fitted.fe_mean,
        fitted.fe_sd,
        strict=True,
    ):
        numeric_estimate = float(estimate)
        numeric_standard_error = float(standard_error)
        z_value = (
            numeric_estimate / numeric_standard_error
            if numeric_standard_error > 0
            else math.nan
        )
        p_value = math.erfc(abs(z_value) / math.sqrt(2.0)) if math.isfinite(z_value) else math.nan
        if not all(
            math.isfinite(value)
            for value in (
                numeric_estimate,
                numeric_standard_error,
                z_value,
                p_value,
            )
        ):
            raise RuntimeError(
                f"mixed-effects logistic fit returned non-finite coefficient {term!r}"
            )
        coefficients.append(
            GEECoefficient(
                term=str(term),
                estimate=numeric_estimate,
                standard_error=numeric_standard_error,
                z_value=z_value,
                p_value=p_value,
            )
        )
    return MixedLogitResult(
        formula=MIXED_EFFECTS_FORMULA,
        group_column=group_column,
        n=len(frame),
        coefficients=tuple(coefficients),
        fit_method="statsmodels.BinomialBayesMixedGLM.fit_vb",
        converged=converged,
        warnings=warning_messages,
    )


def fit_flip_gee(
    rows: Sequence[Mapping[str, Any]],
    *,
    formula: str = FIXED_EFFECTS_FORMULA,
    group_column: str = "question_id",
) -> GEEResult:
    """Fit a question-clustered binomial GEE for the predeclared RQ3 formula."""

    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise OptionalAnalysisDependencyError(
            "fit_flip_gee requires optional dependency 'statsmodels'"
        ) from exc

    frame = _modeling_frame(rows, group_column=group_column)
    model = smf.gee(
        formula=formula,
        groups=frame[group_column],
        data=frame,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    fitted = model.fit()
    coefficients = tuple(
        GEECoefficient(
            term=str(term),
            estimate=float(fitted.params[term]),
            standard_error=float(fitted.bse[term]),
            z_value=float(fitted.tvalues[term]),
            p_value=float(fitted.pvalues[term]),
        )
        for term in fitted.params.index
    )
    return GEEResult(
        formula=formula,
        group_column=group_column,
        n=len(frame),
        coefficients=coefficients,
    )


def fit_uncertainty_gee(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_column: str = "question_id",
) -> UncertaintyGEEResult:
    """Fit Gaussian GEE for uncertainty over canonical normalized dose."""

    try:
        import pandas as pd
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError as exc:  # pragma: no cover - optional environment
        raise OptionalAnalysisDependencyError(
            "fit_uncertainty_gee requires optional dependency 'statsmodels'"
        ) from exc
    frame = pd.DataFrame([dict(row) for row in rows])
    required = {"uncertainty", "normalized_dose", group_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"uncertainty GEE input is missing columns: {missing}")
    frame = frame.dropna(subset=list(required)).copy()
    if frame.empty:
        raise ValueError("uncertainty GEE input has no complete rows")
    n_clusters = int(frame[group_column].nunique())
    if n_clusters < 2:
        raise ValueError("uncertainty GEE requires at least two question clusters")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fitted = smf.gee(
            formula=UNCERTAINTY_GEE_FORMULA,
            groups=frame[group_column],
            data=frame,
            family=sm.families.Gaussian(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
    slope = float(fitted.params["normalized_dose"])
    standard_error = float(fitted.bse["normalized_dose"])
    z_value = slope / standard_error if standard_error > 0.0 else math.nan
    p_value = (
        0.5 * math.erfc(z_value / math.sqrt(2.0))
        if math.isfinite(z_value)
        else math.nan
    )
    warning_messages = tuple(
        dict.fromkeys(str(item.message) for item in captured)
    )
    diagnostics = "; ".join(warning_messages[:3])
    converged = bool(getattr(fitted, "converged", True))
    estimates = (
        float(fitted.params["Intercept"]),
        slope,
        standard_error,
        z_value,
        p_value,
    )
    if not converged:
        raise RuntimeError(
            "uncertainty GEE did not converge"
            + (f": {diagnostics}" if diagnostics else "")
        )
    if standard_error <= 0.0 or not all(math.isfinite(value) for value in estimates):
        raise RuntimeError(
            "uncertainty GEE returned non-finite or degenerate estimates"
            + (f": {diagnostics}" if diagnostics else "")
        )
    return UncertaintyGEEResult(
        formula=UNCERTAINTY_GEE_FORMULA,
        group_column=group_column,
        n=len(frame),
        n_clusters=n_clusters,
        intercept=estimates[0],
        slope=slope,
        slope_standard_error=standard_error,
        slope_z_value=z_value,
        slope_p_value_one_sided=p_value,
        converged=converged,
        warnings=warning_messages,
    )
