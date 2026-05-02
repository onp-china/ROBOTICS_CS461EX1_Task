#!/usr/bin/env python3
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO = Path("/Users/kiki/Documents/New project/course-project-ex1-team-2")
OUT = Path("/Users/kiki/Documents/New project/report_visuals")


def git_lines(args):
    result = subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def shell_lines(command):
    result = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
        shell=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def setup_style():
    plt.style.use("ggplot")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_contributors(contributors):
    top = contributors[:8]
    names = [name for _, name in top][::-1]
    counts = [count for count, _ in top][::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names, counts, color="#2D6A4F")
    ax.set_title("Top Contributors by Commit Count")
    ax.set_xlabel("Commits")
    for idx, value in enumerate(counts):
        ax.text(value + 0.5, idx, str(value), va="center")
    save_fig(OUT / "contributors_top8.png")


def plot_weekly_activity(commit_dates):
    series = pd.Series(1, index=pd.to_datetime(commit_dates))
    weekly = series.resample("W").sum()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(weekly.index, weekly.values, marker="o", linewidth=2.5, color="#1D3557")
    ax.fill_between(weekly.index, weekly.values, alpha=0.18, color="#457B9D")
    ax.set_title("Weekly Commit Activity")
    ax.set_xlabel("Week")
    ax.set_ylabel("Commits")
    save_fig(OUT / "weekly_commit_activity.png")


def plot_commit_types(subjects):
    buckets = Counter()
    for subject in subjects:
        lowered = subject.lower()
        if lowered.startswith("merge"):
            buckets["merge"] += 1
            continue
        match = re.match(r"([a-zA-Z]+)(\(.+\))?:", subject)
        if match:
            buckets[match.group(1).lower()] += 1
        else:
            buckets["other"] += 1

    labels = list(buckets.keys())
    values = list(buckets.values())
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ["#E76F51", "#2A9D8F", "#264653", "#F4A261", "#E9C46A", "#6D597A"]
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors[: len(values)])
    ax.set_title("Commit Message Type Distribution")
    save_fig(OUT / "commit_types_pie.png")


def plot_engineering_snapshot(metrics):
    labels = [
        "All Commits",
        "Main Commits",
        "Merge Commits",
        "Contributors",
        "Remote Branches",
        "PR Merge Msgs",
    ]
    values = [
        metrics["all_commits"],
        metrics["main_commits"],
        metrics["merge_commits"],
        metrics["contributors"],
        metrics["remote_branches"],
        metrics["pr_merge_messages"],
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, values, color=["#0F4C5C", "#1B9AAA", "#EF476F", "#FFC43D", "#6A4C93", "#2D6A4F"])
    ax.set_title("Engineering Snapshot")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, str(value), ha="center")
    save_fig(OUT / "engineering_snapshot.png")


def plot_quality_signals(metrics):
    labels = ["Workflow Files", "Test Dirs", "README Test Mentions", "README CI/CD Mentions"]
    values = [
        metrics["workflow_files"],
        metrics["test_dirs"],
        metrics["readme_test_mentions"],
        metrics["readme_ci_mentions"],
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=["#8D99AE", "#8D99AE", "#2B9348", "#2B9348"])
    ax.set_title("Quality and Automation Signals")
    ax.set_ylabel("Count")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.1, str(value), ha="center")
    save_fig(OUT / "quality_signals.png")


def main():
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)

    contributors_raw = git_lines(["shortlog", "-sn", "--all"])
    contributors = []
    for line in contributors_raw:
        count, name = line.strip().split("\t", 1)
        contributors.append((int(count.strip()), name.strip()))

    commit_dates = git_lines(["log", "--all", "--date=short", "--pretty=format:%ad"])
    subjects = git_lines(["log", "--all", "--pretty=format:%s"])
    merge_subjects = git_lines(["log", "--all", "--merges", "--pretty=format:%s"])
    branches = git_lines(["branch", "-r"])

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    workflow_files = list((REPO / ".github" / "workflows").glob("*")) if (REPO / ".github" / "workflows").exists() else []
    test_dirs = shell_lines(
        "find '/Users/kiki/Documents/New project/course-project-ex1-team-2' -type d \\( -iname 'test' -o -iname 'tests' -o -iname '*gut*' \\)"
    )

    metrics = {
        "all_commits": int(git_lines(["rev-list", "--count", "--all"])[0]),
        "main_commits": int(git_lines(["rev-list", "--count", "main"])[0]),
        "merge_commits": len(git_lines(["log", "--all", "--merges", "--pretty=format:%H"])),
        "contributors": len(contributors),
        "remote_branches": len(branches),
        "pr_merge_messages": sum(1 for subject in merge_subjects if "pull request #" in subject.lower()),
        "workflow_files": len(workflow_files),
        "test_dirs": len(test_dirs),
        "readme_test_mentions": len(re.findall(r"\btest|gut\b", readme, flags=re.IGNORECASE)),
        "readme_ci_mentions": len(re.findall(r"\bci/cd\b|github actions|continuous integration|continuous delivery", readme, flags=re.IGNORECASE)),
    }

    plot_contributors(contributors)
    plot_weekly_activity(commit_dates)
    plot_commit_types(subjects)
    plot_engineering_snapshot(metrics)
    plot_quality_signals(metrics)

    summary = {
        "repo": "BSAI301/course-project-ex1-team-2",
        "metrics": metrics,
        "top_contributors": [{"name": name, "commits": count} for count, name in contributors[:8]],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
