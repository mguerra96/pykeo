"""
Debug helper: plot elevation / STEC / VTEC / dTEC time series for a single station
on a given day (4 stacked subplots sharing the time axis) as an interactive HTML file.

Usage:
    python scripts/plot_station_vtec.py <YYYY-MM-DD> [--save PATH.html]

The script lists all stations available in the parquet for that day and
prompts the user to pick one interactively. Lines are colored per arc id;
hover shows the arc id, clicking a legend entry toggles the arc across all
four panels.
"""

import argparse
import tempfile
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TEC_DIR      = _PROJECT_ROOT / "results" / "tec_data"


def load_day(date_str: str) -> pd.DataFrame:
    parquet = _TEC_DIR / f"tec_{date_str}.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet}")
    return pd.read_parquet(parquet)


def list_stations(df: pd.DataFrame) -> list[str]:
    codes = df["id_arc_valid"].dropna().str.split("_").str[0].str.upper().unique()
    return sorted(codes)


def prompt_station(stations: list[str]) -> list[str]:
    cols = 8
    width = max(len(s) for s in stations) + 2
    print(f"\nStations available ({len(stations)}):")
    for i, s in enumerate(stations):
        end = "\n" if (i + 1) % cols == 0 else ""
        print(f"{i+1:3d}) {s:<{width}}", end=end)
    if len(stations) % cols != 0:
        print()

    while True:
        raw = input("\nPick station(s) — numbers or codes, space/comma separated: ").strip()
        if not raw:
            continue
        tokens = [t for t in raw.replace(",", " ").split() if t]
        picked: list[str] = []
        bad: list[str] = []
        for tok in tokens:
            if tok.isdigit():
                idx = int(tok) - 1
                if 0 <= idx < len(stations):
                    picked.append(stations[idx])
                    continue
            else:
                up = tok.upper()
                if up in stations:
                    picked.append(up)
                    continue
            bad.append(tok)
        if bad:
            print(f"Invalid choice(s): {bad}")
            continue
        seen: set[str] = set()
        unique = [s for s in picked if not (s in seen or seen.add(s))]
        return unique


def filter_station(df: pd.DataFrame, station: str) -> pd.DataFrame:
    station_up = station.upper()
    prefix = df["id_arc_valid"].str.split("_").str[0].str.upper()
    sub = df.loc[prefix == station_up].copy()
    if sub.empty:
        raise ValueError(f"No rows for station '{station_up}'.")
    return sub.sort_values(["id_arc_valid", "epoch"])


_PANELS = [
    ("ele",             "Elevation (deg)"),
    ("stec",            "STEC (TECU)"),
    ("vtec",            "VTEC (TECU)"),
    ("vtec_detrended",  "dTEC (TECU)"),
]

# Plotly "Light24" palette — 24 distinguishable colors, good for many arcs
_PALETTE = [
    "#FD3216", "#00FE35", "#6A76FC", "#FED4C4", "#FE00CE", "#0DF9FF",
    "#F6F926", "#FF9616", "#479B55", "#EEA6FB", "#DC587D", "#D626FF",
    "#6E899C", "#00B5F7", "#B68E00", "#C9FBE5", "#FF0092", "#22FFA7",
    "#E3EE9E", "#86CE00", "#BC7196", "#7E7DCD", "#FC6955", "#E48F72",
]


def plot_html(df: pd.DataFrame, station: str, date_str: str) -> None:
    missing = [c for c, _ in _PANELS if c not in df.columns]
    if missing:
        raise KeyError(f"Columns missing in parquet: {missing}")

    arc_ids = list(df["id_arc_valid"].dropna().unique())
    color_map = {aid: _PALETTE[i % len(_PALETTE)] for i, aid in enumerate(arc_ids)}

    fig = make_subplots(
        rows=len(_PANELS), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[label for _, label in _PANELS],
    )

    for arc_id, g in df.groupby("id_arc_valid", sort=False):
        color = color_map[arc_id]
        for row, (col, _label) in enumerate(_PANELS, start=1):
            fig.add_trace(
                go.Scattergl(
                    x=g["epoch"], y=g[col],
                    mode="markers",
                    name=arc_id,
                    legendgroup=arc_id,
                    showlegend=(row == 1),
                    marker=dict(color=color, size=3),
                    hovertemplate=f"<b>{arc_id}</b><br>%{{x}}<br>%{{y:.3f}}<extra></extra>",
                ),
                row=row, col=1,
            )

    for row, (_col, ylabel) in enumerate(_PANELS, start=1):
        fig.update_yaxes(title_text=ylabel, row=row, col=1, gridcolor="lightgray")
        fig.update_xaxes(gridcolor="lightgray", row=row, col=1)

    fig.update_xaxes(title_text="Time (UTC)", row=len(_PANELS), col=1)

    fig.update_layout(
        title=f"{station.upper()} — {date_str}",
        height=1800,
        hovermode="closest",
        legend=dict(
            title="arc id",
            font=dict(size=9),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        plot_bgcolor="white",
        margin=dict(l=70, r=180, t=70, b=50),
    )

    plot_div = fig.to_html(include_plotlyjs="cdn", full_html=False)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{station.upper()} — {date_str}</title>"
        "<style>html,body{margin:0;padding:0;overflow:auto;}"
        ".plot-wrap{width:100%;height:1800px;}</style>"
        "</head><body>"
        f"<div class='plot-wrap'>{plot_div}</div>"
        "</body></html>"
    )

    tmp = Path(tempfile.mkstemp(prefix=f"vtec_{date_str}_{station.upper()}_", suffix=".html")[1])
    tmp.write_text(html, encoding="utf-8")
    webbrowser.open(tmp.resolve().as_uri())
    print(f"Opened in browser (temp file: {tmp})")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot elevation / STEC / VTEC / dTEC time series for one station as interactive HTML.")
    p.add_argument("date", help="Date YYYY-MM-DD")
    args = p.parse_args()

    df_all = load_day(args.date)
    stations = list_stations(df_all)
    if not stations:
        raise SystemExit(f"No stations found in {args.date} parquet.")

    picks = prompt_station(stations)
    for station in picks:
        df = filter_station(df_all, station)

        print(f"{station} {args.date}: {len(df):,} rows, "
              f"{df['sv'].nunique()} SVs, {df['id_arc_valid'].nunique()} arcs")

        plot_html(df, station, args.date)


if __name__ == "__main__":
    main()
