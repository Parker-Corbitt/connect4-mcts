#!/usr/bin/env python3
import argparse
import ast
import csv
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional


GAME_RE = re.compile(
    r"^game=(?P<idx>\d+)/(?P<total>\d+)\s+llm_side=(?P<side>first|second)\s+winner=(?P<winner>-?1|0)$"
)
KV_RE = re.compile(r"^(?P<key>[a-z_]+)=(?P<value>.+)$")


@dataclass
class GameRecord:
    idx: int
    total: int
    llm_side: str
    winner: int

    @property
    def llm_player(self) -> int:
        return -1 if self.llm_side == "first" else 1

    @property
    def llm_result(self) -> str:
        if self.winner == 0:
            return "draw"
        if self.winner == self.llm_player:
            return "win"
        return "loss"


@dataclass
class RunSummary:
    file: str
    iterations_per_move: int
    games_declared: int
    llm_wins: int
    pure_wins: int
    draws: int
    llm_calls: Optional[int]
    cache_entries: Optional[int]
    cache_hits: Optional[int]
    game_records: List[GameRecord]

    @property
    def llm_win_rate(self) -> float:
        total = self.llm_wins + self.pure_wins + self.draws
        return (self.llm_wins / total) if total else 0.0

    @property
    def llm_non_draw_win_rate(self) -> float:
        decisive = self.llm_wins + self.pure_wins
        return (self.llm_wins / decisive) if decisive else 0.0

    def side_stats(self) -> Dict[str, Dict[str, float]]:
        out = {
            "first": {"games": 0, "wins": 0},
            "second": {"games": 0, "wins": 0},
        }
        for record in self.game_records:
            side = record.llm_side
            out[side]["games"] += 1
            if record.llm_result == "win":
                out[side]["wins"] += 1
        for side in ("first", "second"):
            games = out[side]["games"]
            out[side]["win_rate"] = (out[side]["wins"] / games) if games else 0.0
        return out


def parse_run_file(path: str) -> RunSummary:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    game_records: List[GameRecord] = []
    summary: Dict[str, str] = {}

    for line in lines:
        game_match = GAME_RE.match(line)
        if game_match:
            game_records.append(
                GameRecord(
                    idx=int(game_match.group("idx")),
                    total=int(game_match.group("total")),
                    llm_side=game_match.group("side"),
                    winner=int(game_match.group("winner")),
                )
            )
            continue

        kv_match = KV_RE.match(line)
        if kv_match:
            summary[kv_match.group("key")] = kv_match.group("value")

    if not summary:
        raise ValueError(f"{path}: could not find summary key/value lines")

    iterations_per_move = int(summary.get("iterations_per_move", "0"))
    games_declared = int(summary.get("games", str(len(game_records))))

    llm_wins = int(summary.get("llm_wins", "0"))
    pure_wins = int(summary.get("pure_wins", "0"))
    draws = int(summary.get("draws", "0"))

    stats_raw = summary.get("llm_policy_stats")
    stats = {}
    if stats_raw:
        try:
            parsed = ast.literal_eval(stats_raw)
            if isinstance(parsed, dict):
                stats = parsed
        except (SyntaxError, ValueError):
            stats = {}

    # Backfill from game lines if summary counters are missing or zeroed.
    if game_records and (llm_wins + pure_wins + draws) == 0:
        llm_wins = sum(1 for g in game_records if g.llm_result == "win")
        pure_wins = sum(1 for g in game_records if g.llm_result == "loss")
        draws = sum(1 for g in game_records if g.llm_result == "draw")

    return RunSummary(
        file=os.path.basename(path),
        iterations_per_move=iterations_per_move,
        games_declared=games_declared,
        llm_wins=llm_wins,
        pure_wins=pure_wins,
        draws=draws,
        llm_calls=stats.get("llm_calls"),
        cache_entries=stats.get("cache_entries"),
        cache_hits=stats.get("cache_hits"),
        game_records=game_records,
    )


def write_run_summary_csv(runs: List[RunSummary], out_path: str) -> None:
    fieldnames = [
        "file",
        "iterations_per_move",
        "games",
        "llm_wins",
        "pure_wins",
        "draws",
        "llm_win_rate",
        "llm_non_draw_win_rate",
        "llm_calls",
        "cache_entries",
        "cache_hits",
        "llm_first_games",
        "llm_first_wins",
        "llm_first_win_rate",
        "llm_second_games",
        "llm_second_wins",
        "llm_second_win_rate",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            side = run.side_stats()
            writer.writerow(
                {
                    "file": run.file,
                    "iterations_per_move": run.iterations_per_move,
                    "games": run.games_declared,
                    "llm_wins": run.llm_wins,
                    "pure_wins": run.pure_wins,
                    "draws": run.draws,
                    "llm_win_rate": f"{run.llm_win_rate:.4f}",
                    "llm_non_draw_win_rate": f"{run.llm_non_draw_win_rate:.4f}",
                    "llm_calls": run.llm_calls,
                    "cache_entries": run.cache_entries,
                    "cache_hits": run.cache_hits,
                    "llm_first_games": side["first"]["games"],
                    "llm_first_wins": side["first"]["wins"],
                    "llm_first_win_rate": f"{side['first']['win_rate']:.4f}",
                    "llm_second_games": side["second"]["games"],
                    "llm_second_wins": side["second"]["wins"],
                    "llm_second_win_rate": f"{side['second']['win_rate']:.4f}",
                }
            )


def write_game_level_csv(runs: List[RunSummary], out_path: str) -> None:
    fieldnames = [
        "file",
        "iterations_per_move",
        "game_index",
        "game_total",
        "llm_side",
        "winner",
        "llm_result",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            for rec in run.game_records:
                writer.writerow(
                    {
                        "file": run.file,
                        "iterations_per_move": run.iterations_per_move,
                        "game_index": rec.idx,
                        "game_total": rec.total,
                        "llm_side": rec.llm_side,
                        "winner": rec.winner,
                        "llm_result": rec.llm_result,
                    }
                )


def markdown_table(runs: List[RunSummary]) -> str:
    header = (
        "| file | iters/move | games | LLM wins | Pure wins | draws | LLM win rate | "
        "LLM non-draw win rate | LLM calls | cache entries | cache hits |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = []
    for run in runs:
        rows.append(
            "| {file} | {iters} | {games} | {lw} | {pw} | {d} | {wr:.2%} | {ndr:.2%} | {calls} | {entries} | {hits} |".format(
                file=run.file,
                iters=run.iterations_per_move,
                games=run.games_declared,
                lw=run.llm_wins,
                pw=run.pure_wins,
                d=run.draws,
                wr=run.llm_win_rate,
                ndr=run.llm_non_draw_win_rate,
                calls=run.llm_calls if run.llm_calls is not None else "",
                entries=run.cache_entries if run.cache_entries is not None else "",
                hits=run.cache_hits if run.cache_hits is not None else "",
            )
        )
    return header + "\n" + "\n".join(rows)


def aggregate_by_iterations(runs: List[RunSummary]) -> List[Dict[str, float]]:
    agg: Dict[int, Dict[str, float]] = {}
    for run in runs:
        iters = run.iterations_per_move
        if iters not in agg:
            agg[iters] = {
                "iterations_per_move": iters,
                "runs": 0,
                "games": 0,
                "llm_wins": 0,
                "pure_wins": 0,
                "draws": 0,
                "llm_first_games": 0,
                "llm_first_wins": 0,
                "llm_second_games": 0,
                "llm_second_wins": 0,
            }
        side = run.side_stats()
        agg[iters]["runs"] += 1
        agg[iters]["games"] += run.games_declared
        agg[iters]["llm_wins"] += run.llm_wins
        agg[iters]["pure_wins"] += run.pure_wins
        agg[iters]["draws"] += run.draws
        agg[iters]["llm_first_games"] += side["first"]["games"]
        agg[iters]["llm_first_wins"] += side["first"]["wins"]
        agg[iters]["llm_second_games"] += side["second"]["games"]
        agg[iters]["llm_second_wins"] += side["second"]["wins"]

    rows = []
    for iters in sorted(agg):
        row = agg[iters]
        games = row["games"]
        decisive = row["llm_wins"] + row["pure_wins"]
        row["llm_win_rate"] = (row["llm_wins"] / games) if games else 0.0
        row["llm_non_draw_win_rate"] = (row["llm_wins"] / decisive) if decisive else 0.0

        row["llm_first_win_rate"] = (
            row["llm_first_wins"] / row["llm_first_games"]
            if row["llm_first_games"]
            else 0.0
        )
        row["llm_second_win_rate"] = (
            row["llm_second_wins"] / row["llm_second_games"]
            if row["llm_second_games"]
            else 0.0
        )
        rows.append(row)
    return rows


def write_iteration_summary_csv(rows: List[Dict[str, float]], out_path: str) -> None:
    fieldnames = [
        "iterations_per_move",
        "runs",
        "games",
        "llm_wins",
        "pure_wins",
        "draws",
        "llm_win_rate",
        "llm_non_draw_win_rate",
        "llm_first_win_rate",
        "llm_second_win_rate",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "iterations_per_move": int(row["iterations_per_move"]),
                    "runs": int(row["runs"]),
                    "games": int(row["games"]),
                    "llm_wins": int(row["llm_wins"]),
                    "pure_wins": int(row["pure_wins"]),
                    "draws": int(row["draws"]),
                    "llm_win_rate": f"{row['llm_win_rate']:.4f}",
                    "llm_non_draw_win_rate": f"{row['llm_non_draw_win_rate']:.4f}",
                    "llm_first_win_rate": f"{row['llm_first_win_rate']:.4f}",
                    "llm_second_win_rate": f"{row['llm_second_win_rate']:.4f}",
                }
            )


def write_markdown_summary(
    runs: List[RunSummary], iter_rows: List[Dict[str, float]], out_path: str
) -> None:
    lines = []
    lines.append("# Connect4 Battle Summary")
    lines.append("")
    lines.append("## Per-run table")
    lines.append("")
    lines.append(markdown_table(runs))
    lines.append("")
    lines.append("## Aggregated by iterations")
    lines.append("")
    lines.append(
        "| iters/move | runs | games | LLM wins | Pure wins | draws | LLM win rate | "
        "LLM non-draw win rate | LLM first win rate | LLM second win rate |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in iter_rows:
        lines.append(
            "| {iters} | {runs} | {games} | {lw} | {pw} | {d} | {wr:.2%} | {ndr:.2%} | {fwr:.2%} | {swr:.2%} |".format(
                iters=int(row["iterations_per_move"]),
                runs=int(row["runs"]),
                games=int(row["games"]),
                lw=int(row["llm_wins"]),
                pw=int(row["pure_wins"]),
                d=int(row["draws"]),
                wr=row["llm_win_rate"],
                ndr=row["llm_non_draw_win_rate"],
                fwr=row["llm_first_win_rate"],
                swr=row["llm_second_win_rate"],
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `LLM win rate` uses all games (draws count in denominator).")
    lines.append("- `LLM non-draw win rate` excludes draws.")
    lines.append("- `LLM first/second win rate` tracks side bias.")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def try_write_plots(rows: List[Dict[str, float]], out_dir: str) -> List[str]:
    written = []
    mpl_config = os.path.join(out_dir, ".mplconfig")
    xdg_cache = os.path.join(out_dir, ".cache")
    os.makedirs(mpl_config, exist_ok=True)
    os.makedirs(xdg_cache, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_config)
    os.environ.setdefault("XDG_CACHE_HOME", xdg_cache)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return written

    iters = [int(r["iterations_per_move"]) for r in rows]
    llm_wins = [int(r["llm_wins"]) for r in rows]
    pure_wins = [int(r["pure_wins"]) for r in rows]
    draws = [int(r["draws"]) for r in rows]
    win_rate = [float(r["llm_win_rate"]) for r in rows]
    first_rate = [float(r["llm_first_win_rate"]) for r in rows]
    second_rate = [float(r["llm_second_win_rate"]) for r in rows]

    # 1) Win rate line plot.
    plt.figure(figsize=(9, 5))
    plt.plot(iters, win_rate, marker="o", linewidth=2)
    plt.ylim(0, 1)
    plt.xlabel("Iterations per move")
    plt.ylabel("LLM win rate")
    plt.title("LLM win rate vs iterations")
    plt.grid(alpha=0.3)
    out = os.path.join(out_dir, "llm_win_rate_vs_iterations.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    written.append(out)

    # 2) Outcome counts stacked bar.
    plt.figure(figsize=(9, 5))
    plt.bar(iters, llm_wins, label="LLM wins")
    plt.bar(iters, pure_wins, bottom=llm_wins, label="Pure wins")
    bottoms = [lw + pw for lw, pw in zip(llm_wins, pure_wins)]
    plt.bar(iters, draws, bottom=bottoms, label="Draws")
    plt.xlabel("Iterations per move")
    plt.ylabel("Games")
    plt.title("Outcomes by iterations")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    out = os.path.join(out_dir, "outcomes_by_iterations.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    written.append(out)

    # 3) Side-bias plot.
    plt.figure(figsize=(9, 5))
    plt.plot(iters, first_rate, marker="o", label="LLM as first")
    plt.plot(iters, second_rate, marker="o", label="LLM as second")
    plt.ylim(0, 1)
    plt.xlabel("Iterations per move")
    plt.ylabel("LLM side-specific win rate")
    plt.title("Side bias by iterations")
    plt.legend()
    plt.grid(alpha=0.3)
    out = os.path.join(out_dir, "llm_side_win_rates.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    written.append(out)

    return written


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize connect4 battle outputs into tables and plots."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["output*.txt"],
        help="Input files or glob patterns (default: output*.txt)",
    )
    parser.add_argument(
        "--out-dir",
        default="analysis",
        help="Output directory for csv/markdown/plots",
    )
    return parser.parse_args()


def expand_inputs(patterns: List[str]) -> List[str]:
    paths = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(matches)
        elif os.path.isfile(pattern):
            paths.append(pattern)
    unique = sorted(set(paths))
    return unique


def main():
    args = parse_args()
    input_paths = expand_inputs(args.inputs)
    if not input_paths:
        raise SystemExit("No input files found.")

    os.makedirs(args.out_dir, exist_ok=True)

    runs = [parse_run_file(path) for path in input_paths]
    runs.sort(key=lambda r: (r.iterations_per_move, r.file))
    iter_rows = aggregate_by_iterations(runs)

    run_summary_csv = os.path.join(args.out_dir, "run_summary.csv")
    game_level_csv = os.path.join(args.out_dir, "game_level.csv")
    iteration_summary_csv = os.path.join(args.out_dir, "iteration_summary.csv")
    summary_md = os.path.join(args.out_dir, "summary.md")

    write_run_summary_csv(runs, run_summary_csv)
    write_game_level_csv(runs, game_level_csv)
    write_iteration_summary_csv(iter_rows, iteration_summary_csv)
    write_markdown_summary(runs, iter_rows, summary_md)
    plot_files = try_write_plots(iter_rows, args.out_dir)

    print(f"parsed_runs={len(runs)}")
    print(f"wrote={run_summary_csv}")
    print(f"wrote={game_level_csv}")
    print(f"wrote={iteration_summary_csv}")
    print(f"wrote={summary_md}")
    if plot_files:
        for path in plot_files:
            print(f"wrote={path}")
    else:
        print("plots=skipped (matplotlib not installed)")

    print("\nPer-run table:\n")
    print(markdown_table(runs))


if __name__ == "__main__":
    main()
