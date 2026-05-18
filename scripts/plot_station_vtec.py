"""
Debug helper: plot elevation / STEC / VTEC / dTEC time series for a single station
on a given day (4 stacked subplots sharing the time axis).

Usage:
    python scripts/plot_station_vtec.py <YYYY-MM-DD> [--save PATH]

The script lists all stations available in the parquet for that day and
prompts the user to pick one interactively. Lines are colored per arc id;
hover shows the arc id, clicking a legend entry toggles the arc across all
three panels.
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TEC_DIR      = _PROJECT_ROOT / "results" / "tec_data"


def load_day(date_str: str) -> pd.DataFrame:
    parquet = _TEC_DIR / f"tec_{date_str}.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet}")
    return pd.read_parquet(parquet)


def list_stations(df: pd.DataFrame) -> list[str]:
    # Station code is the first underscore-delimited token of id_arc_valid
    # (e.g. "mate_G05_0" or "MATE_G05_0"). Normalize to upper-case for display.
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
        # de-dup while preserving order
        seen: set[str] = set()
        unique = [s for s in picked if not (s in seen or seen.add(s))]
        return unique


def filter_station(df: pd.DataFrame, station: str) -> pd.DataFrame:
    station_up = station.upper()
    # id_arc_valid may use upper- or lower-case station codes — match case-insensitively.
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


def plot(df: pd.DataFrame, station: str, date_str: str, save: Path | None) -> None:
    missing = [c for c, _ in _PANELS if c not in df.columns]
    if missing:
        raise KeyError(f"Columns missing in parquet: {missing}")

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    arc_ids = list(df["id_arc_valid"].dropna().unique())
    cmap = plt.get_cmap("tab20")
    color_map = {aid: cmap(i % cmap.N) for i, aid in enumerate(arc_ids)}

    # arc_id -> list of Line2D (one per panel, same order as _PANELS)
    arc_lines: dict[str, list] = {aid: [] for aid in arc_ids}

    for arc_id, g in df.groupby("id_arc_valid", sort=False):
        color = color_map[arc_id]
        for ax, (col, _label) in zip(axes, _PANELS):
            (line,) = ax.plot(
                g["epoch"], g[col],
                color=color, linewidth=1.0, alpha=0.85,
                label=arc_id, picker=5,
            )
            arc_lines[arc_id].append(line)

    for ax, (_col, ylabel) in zip(axes, _PANELS):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(f"{station.upper()} — {date_str}")
    axes[-1].set_xlabel("Time (UTC)")

    # Single legend on the right of the figure, shared across panels
    legend_handles = [arc_lines[aid][0] for aid in arc_ids]
    leg = fig.legend(
        legend_handles, arc_ids,
        loc="center left", bbox_to_anchor=(0.88, 0.5),
        fontsize=7, ncol=1, frameon=False,
    )

    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 0.87, 1))

    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved: {save}")
        return

    _attach_hover_tooltip(fig, axes, arc_lines)
    _attach_legend_toggle(fig, leg, arc_lines)
    print("Interactive: hover for arc id, click legend entry to toggle visibility.")
    plt.show()


def _attach_hover_tooltip(fig, axes, arc_lines: dict) -> None:
    """Per-axes annotation showing the arc_id of the line under the cursor."""
    annots = {}
    for ax in axes:
        ann = ax.annotate(
            "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.95),
            fontsize=8, zorder=10,
        )
        ann.set_visible(False)
        annots[ax] = ann

    # Reverse index: per-axis dict of arc_id -> Line2D for fast hit testing
    axis_arc_lines = {ax: {} for ax in axes}
    for aid, lines in arc_lines.items():
        for ax, line in zip(axes, lines):
            axis_arc_lines[ax][aid] = line

    def hide_all():
        changed = False
        for ann in annots.values():
            if ann.get_visible():
                ann.set_visible(False)
                changed = True
        if changed:
            fig.canvas.draw_idle()

    def on_move(event):
        ax = event.inaxes
        if ax not in annots:
            hide_all()
            return
        hit_label = None
        for label, line in axis_arc_lines[ax].items():
            if not line.get_visible():
                continue
            contains, _ = line.contains(event)
            if contains:
                hit_label = label
                break
        # hide all other panels' tooltips
        for other_ax, ann in annots.items():
            if other_ax is not ax and ann.get_visible():
                ann.set_visible(False)
        ann = annots[ax]
        if hit_label is None:
            if ann.get_visible():
                ann.set_visible(False)
                fig.canvas.draw_idle()
            return
        ann.xy = (event.xdata, event.ydata)
        ann.set_text(hit_label)
        ann.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)


def _attach_legend_toggle(fig, leg, arc_lines: dict) -> None:
    """Click a legend entry to hide/show the arc across all panels."""
    legline_to_label = {}
    for legline, label in zip(leg.get_lines(), arc_lines.keys()):
        legline.set_picker(True)
        legline.set_pickradius(5)
        legline_to_label[legline] = label

    def on_pick(event):
        legline = event.artist
        label = legline_to_label.get(legline)
        if label is None:
            return
        lines = arc_lines[label]
        visible = not lines[0].get_visible()
        for line in lines:
            line.set_visible(visible)
        legline.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot elevation / STEC / VTEC time series for one station on one day.")
    p.add_argument("date", help="Date YYYY-MM-DD")
    p.add_argument("--save", type=Path, default=None, help="Output image path (otherwise show interactively)")
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

        save_path: Path | None = args.save
        if save_path is not None and len(picks) > 1:
            save_path = save_path.with_name(f"{save_path.stem}_{station}{save_path.suffix}")
        plot(df, station, args.date, save_path)


if __name__ == "__main__":
    main()
