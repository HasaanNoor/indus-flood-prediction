from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from paths import PROJECT_ROOT, VALIDATION_DIR


DEFAULT_REPORT_PATH = VALIDATION_DIR / "preprocessing_report.txt"


def _time_values(ds: xr.Dataset) -> pd.DatetimeIndex | None:
    if "time" not in ds.coords and "time" not in ds.dims:
        return None
    return pd.DatetimeIndex(pd.to_datetime(ds["time"].values)).sort_values()


def _format_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def describe_dataset(path: Path, name: str) -> list[str]:
    lines = [f"[{name}]", f"path: {_format_relative(path)}"]
    if not path.exists():
        return [*lines, "status: missing", ""]

    ds = xr.open_dataset(path, engine="netcdf4")
    try:
        lines.append(f"dimensions: {dict(ds.sizes)}")
        lines.append(f"variables: {list(ds.data_vars)}")

        times = _time_values(ds)
        if times is None or len(times) == 0:
            lines.append("time: missing")
        else:
            duplicates = int(times.duplicated().sum())
            diffs = times.to_series().diff().dropna()
            expected = pd.Timedelta(days=1)
            missing_dates = pd.date_range(times.min(), times.max(), freq="D").difference(times)
            irregular = int((diffs != expected).sum()) if len(diffs) else 0

            lines.extend(
                [
                    f"min_date: {times.min().date()}",
                    f"max_date: {times.max().date()}",
                    f"time_steps: {len(times)}",
                    f"duplicate_timestamps: {duplicates}",
                    f"missing_daily_dates: {len(missing_dates)}",
                    f"irregular_daily_spacing_count: {irregular}",
                ]
            )
            if len(missing_dates):
                preview = ", ".join(str(date.date()) for date in missing_dates[:10])
                suffix = " ..." if len(missing_dates) > 10 else ""
                lines.append(f"missing_daily_dates_preview: {preview}{suffix}")

        lines.append("variable_summaries:")
        for var_name, data in ds.data_vars.items():
            if var_name == "spatial_ref":
                continue
            values = data
            numeric = np.issubdtype(values.dtype, np.number)
            nan_count = int(values.isnull().sum().item()) if numeric else "n/a"
            if numeric:
                min_value = values.min(skipna=True).item()
                max_value = values.max(skipna=True).item()
                mean_value = values.mean(skipna=True).item()
                lines.append(
                    f"  - {var_name}: dims={values.dims}, dtype={values.dtype}, "
                    f"nan_count={nan_count}, min={min_value:.6g}, "
                    f"max={max_value:.6g}, mean={mean_value:.6g}"
                )
            else:
                lines.append(
                    f"  - {var_name}: dims={values.dims}, dtype={values.dtype}, nan_count={nan_count}"
                )
    finally:
        ds.close()

    lines.append("")
    return lines


def write_preprocessing_report(
    datasets: dict[str, Path],
    output_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Preprocessing Validation Report",
        f"generated_at_utc: {pd.Timestamp.utcnow().isoformat()}",
        "",
    ]
    for name, path in datasets.items():
        lines.extend(describe_dataset(path, name))

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nValidation report saved: {output_path.relative_to(PROJECT_ROOT)}")
    return output_path
