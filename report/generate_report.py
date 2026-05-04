"""Generate the VecScaleDB ANN benchmark comparison PDF.

Run from the project root after collecting metrics:

    python report/collect_metrics.py
    python report/generate_report.py

The PDF writer is deliberately small and self-contained. It draws text, tables,
and vector charts directly using PDF drawing commands.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import wrap
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
BUILD_DIR = REPORT_DIR / "build"
METRICS_PATH = BUILD_DIR / "metrics.json"
OUTPUT_PATH = REPORT_DIR / "vecscaledb_ann_benchmark_report.pdf"

PAGE_W = 595.0
PAGE_H = 842.0
MARGIN = 48.0

BLACK = (0, 0, 0)
WHITE = (1, 1, 1)
BLUE = (0.10, 0.27, 0.48)
ORANGE = (0.85, 0.36, 0.10)
GREEN = (0.20, 0.55, 0.32)
RED = (0.76, 0.20, 0.20)
DARK_GRAY = (0.25, 0.25, 0.25)
LIGHT_GRAY = (0.78, 0.82, 0.86)
COLORS = [BLUE, ORANGE, GREEN]


@dataclass
class Page:
    commands: list[str] = field(default_factory=list)


class Pdf:
    def __init__(self) -> None:
        self.pages: list[Page] = []
        self.page: Page | None = None

    def add_page(self) -> None:
        self.page = Page()
        self.pages.append(self.page)

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: float = 10,
        bold: bool = False,
        color: tuple[float, float, float] = (0, 0, 0),
    ) -> None:
        self._ensure_page()
        font = "F2" if bold else "F1"
        r, g, b = color
        self.page.commands.append(f"{r:.3f} {g:.3f} {b:.3f} rg")
        self.page.commands.append(
            f"BT /{font} {size:.2f} Tf {x:.2f} {self._py(y):.2f} Td ({_esc(value)}) Tj ET"
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: tuple[float, float, float] = (0, 0, 0),
        width: float = 1,
    ) -> None:
        self._ensure_page()
        r, g, b = color
        self.page.commands.append(f"{r:.3f} {g:.3f} {b:.3f} RG")
        self.page.commands.append(f"{width:.2f} w")
        self.page.commands.append(
            f"{x1:.2f} {self._py(y1):.2f} m {x2:.2f} {self._py(y2):.2f} l S"
        )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: tuple[float, float, float] | None = None,
        stroke: tuple[float, float, float] | None = None,
        width: float = 1,
    ) -> None:
        self._ensure_page()
        if fill is not None:
            r, g, b = fill
            self.page.commands.append(f"{r:.3f} {g:.3f} {b:.3f} rg")
        if stroke is not None:
            r, g, b = stroke
            self.page.commands.append(f"{r:.3f} {g:.3f} {b:.3f} RG")
            self.page.commands.append(f"{width:.2f} w")
        op = "B" if fill is not None and stroke is not None else "f" if fill else "S"
        self.page.commands.append(
            f"{x:.2f} {self._py(y + h):.2f} {w:.2f} {h:.2f} re {op}"
        )

    def polyline(
        self,
        points: list[tuple[float, float]],
        color: tuple[float, float, float],
        width: float = 1.5,
    ) -> None:
        if len(points) < 2:
            return
        self._ensure_page()
        r, g, b = color
        parts = [f"{r:.3f} {g:.3f} {b:.3f} RG", f"{width:.2f} w"]
        x0, y0 = points[0]
        parts.append(f"{x0:.2f} {self._py(y0):.2f} m")
        for x, y in points[1:]:
            parts.append(f"{x:.2f} {self._py(y):.2f} l")
        parts.append("S")
        self.page.commands.extend(parts)

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        fill: tuple[float, float, float],
    ) -> None:
        self._ensure_page()
        r, g, b = fill
        k = 0.5522847498
        c = radius * k
        yp = self._py(y)
        cmds = [
            f"{r:.3f} {g:.3f} {b:.3f} rg",
            f"{x + radius:.2f} {yp:.2f} m",
            f"{x + radius:.2f} {yp + c:.2f} {x + c:.2f} {yp + radius:.2f} {x:.2f} {yp + radius:.2f} c",
            f"{x - c:.2f} {yp + radius:.2f} {x - radius:.2f} {yp + c:.2f} {x - radius:.2f} {yp:.2f} c",
            f"{x - radius:.2f} {yp - c:.2f} {x - c:.2f} {yp - radius:.2f} {x:.2f} {yp - radius:.2f} c",
            f"{x + c:.2f} {yp - radius:.2f} {x + radius:.2f} {yp - c:.2f} {x + radius:.2f} {yp:.2f} c",
            "f",
        ]
        self.page.commands.extend(cmds)

    def save(self, path: Path) -> None:
        objects: list[bytes] = []
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        objects.append(b"")
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

        page_refs = []
        for page in self.pages:
            content = "\n".join(page.commands).encode("latin-1", errors="replace")
            stream = (
                f"<< /Length {len(content)} >>\nstream\n".encode("latin-1")
                + content
                + b"\nendstream"
            )
            content_obj = len(objects) + 1
            objects.append(stream)
            page_obj = len(objects) + 1
            page_refs.append(page_obj)
            objects.append(
                (
                    "<< /Type /Page /Parent 2 0 R "
                    f"/MediaBox [0 0 {PAGE_W:.0f} {PAGE_H:.0f}] "
                    "/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
                    f"/Contents {content_obj} 0 R >>"
                ).encode("latin-1")
            )

        kids = " ".join(f"{ref} 0 R" for ref in page_refs)
        objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_refs)} >>".encode(
            "latin-1"
        )

        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for i, obj in enumerate(objects, start=1):
            offsets.append(len(out))
            out.extend(f"{i} 0 obj\n".encode("latin-1"))
            out.extend(obj)
            out.extend(b"\nendobj\n")

        xref_offset = len(out)
        out.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
        out.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            out.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
        out.extend(
            (
                "trailer\n"
                f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                "startxref\n"
                f"{xref_offset}\n"
                "%%EOF\n"
            ).encode("latin-1")
        )
        path.write_bytes(bytes(out))

    def _ensure_page(self) -> None:
        if self.page is None:
            self.add_page()

    @staticmethod
    def _py(y: float) -> float:
        return PAGE_H - y


def main() -> None:
    os.chdir(ROOT)
    if not METRICS_PATH.exists():
        raise SystemExit(
            "Missing report/build/metrics.json. Run python report/collect_metrics.py first."
        )
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    pdf = Pdf()
    build_report(pdf, metrics)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")


def build_report(pdf: Pdf, metrics: dict[str, Any]) -> None:
    datasets = metrics["datasets"]
    benchmarks = metrics["benchmarks"]
    scaling = benchmarks["scaling"]
    sift1m_scaling = scaling.get("SIFT1M", [])

    cover_page(pdf, metrics)
    dataset_page(pdf, datasets)
    architecture_page(pdf)
    scaling_page(pdf, sift1m_scaling)
    comparison_page(pdf, datasets, benchmarks)
    methodology_page(pdf, metrics)


def cover_page(pdf: Pdf, metrics: dict[str, Any]) -> None:
    pdf.add_page()
    y = 70
    pdf.text(MARGIN, y, "VecScaleDB ANN Benchmark Report", 24, bold=True, color=BLUE)
    y += 28
    pdf.text(MARGIN, y, "Distributed Vector Database Evaluation", 14, bold=True)
    y += 30
    y = paragraph(
        pdf,
        MARGIN,
        y,
        "This report compares VecScaleDB against the standard ANN benchmark datasets "
        "used in the Milvus evaluation setup: SIFT10K, SIFT1M, and Deep1M. The focus "
        "is the Figure 10b-style QPS-vs-nodes experiment, with secondary metrics for "
        "latency, errors, data coverage, and benchmark readiness.",
        width=92,
    )
    y += 16
    pdf.text(MARGIN, y, "Generated from local workspace metrics", 11, bold=True)
    y += 16
    pdf.text(MARGIN, y, f"Timestamp: {metrics.get('generated_at', 'unknown')}", 9)
    y += 30

    rows = [
        ("System", "VecScaleDB"),
        ("Repository", "https://github.com/sanchit-22/Distributed-Vector-DB"),
        ("Team", "Deadlock and Chill"),
        ("Member", "Sanchit Kumar - 2024201042"),
        ("Member", "Priyanshu Jha - 2024201062"),
        ("Member", "Rohini Mandlekar - 2025901004"),
        ("Index", "Faiss IVF_FLAT"),
        ("Distance metric", "L2"),
        ("Primary scaling artifact", "QPS vs reader nodes"),
        ("Reader modes", "Replicated round-robin and scatter-gather"),
        ("Durability", "WAL + MemTable + immutable segments"),
    ]
    y = key_value_table(pdf, MARGIN, y, rows, 190, 270)

    y += 26
    pdf.text(MARGIN, y, "Report contents", 14, bold=True, color=BLUE)
    y += 18
    for item in [
        "Dataset specification and local availability",
        "Project architecture overview",
        "SIFT1M QPS and latency charts from local scaling_results.json",
        "Cross-dataset comparison framework for SIFT10K, SIFT1M, and Deep1M",
        "Methodology and commands to regenerate missing measurements",
    ]:
        y = bullet(pdf, MARGIN + 10, y, item)


def dataset_page(pdf: Pdf, datasets: list[dict[str, Any]]) -> None:
    pdf.add_page()
    y = page_title(pdf, "Dataset Comparison")
    y = paragraph(
        pdf,
        MARGIN,
        y,
        "All three datasets are public ANN benchmarks. SIFT10K is used for fast "
        "correctness and recall checks, SIFT1M is the primary local performance "
        "benchmark, and Deep1M is included as the paper-replication target when the "
        "96-dimensional dataset is supplied locally.",
        width=92,
    )
    y += 16

    headers = ["Dataset", "Vectors", "Dims", "Metric", "Available", "Role"]
    rows = []
    for d in datasets:
        rows.append(
            [
                d["name"],
                f'{d["expected_vectors"]:,}',
                str(d["expected_dims"]),
                d["metric"],
                "yes" if d["available"] else "no",
                d["role"],
            ]
        )
    y = table(pdf, MARGIN, y, headers, rows, [75, 70, 45, 50, 65, 220])

    y += 20
    chart_dataset_sizes(pdf, MARGIN, y, 230, 210, datasets)
    chart_dataset_dims(pdf, MARGIN + 275, y, 220, 210, datasets)

    y += 245
    pdf.text(MARGIN, y, "Local dataset inspection", 13, bold=True, color=BLUE)
    y += 17
    for d in datasets:
        status = "available" if d["available"] else "missing"
        detail = f'{d["name"]}: {status}; path={d["path"]}'
        if d["available"]:
            detail += f'; size={d["size_mb"]} MB'
        y = bullet(pdf, MARGIN + 10, y, detail, size=9)
        keys = d.get("hdf5_keys", {})
        if keys:
            for key, info in keys.items():
                y = bullet(
                    pdf,
                    MARGIN + 24,
                    y,
                    f'{key}: shape={info["shape"]}, dtype={info["dtype"]}',
                    size=8,
                    color=DARK_GRAY,
                )


def architecture_page(pdf: Pdf) -> None:
    pdf.add_page()
    y = page_title(pdf, "Project Overview")
    y = paragraph(
        pdf,
        MARGIN,
        y,
        "VecScaleDB uses a writer/readers/coordinator split. Inserts go to the writer, "
        "which appends to the WAL, stores recent writes in a MemTable, and flushes "
        "immutable Faiss-backed segments to shared storage. Readers load those "
        "segments and answer search requests. Coordinators store metadata in etcd and "
        "route search requests.",
        width=92,
    )
    y += 16

    boxes = [
        ("Client", "Insert/search requests"),
        ("Writer", "WAL, MemTable, segment flush"),
        ("Shared storage", "index.faiss, ids.npy, vectors.npy, meta.json"),
        ("Coordinator", "Node registry, segment metadata, routing"),
        ("Readers", "Segment cache and IVF_FLAT search"),
        ("etcd", "Leader, nodes, segments, snapshot"),
    ]
    flow_chart(pdf, MARGIN, y, boxes)

    y += 230
    pdf.text(MARGIN, y, "Feature coverage", 13, bold=True, color=BLUE)
    y += 18
    features = [
        ("Distributed architecture", "3 coordinators, 1 writer, 5 readers, shared storage, etcd"),
        ("Load balancing", "replicated readers receive coordinator round-robin search traffic"),
        ("Vector search", "Faiss IVF_FLAT with configurable nprobe"),
        ("Dynamic data", "insert/delete through WAL-backed writer pipeline"),
        ("Snapshot isolation", "search uses snapshot_id derived from WAL/segment LSNs"),
        ("Fault tolerance", "readers reload segments; writer replays WAL; coordinator leader failover"),
    ]
    y = table(pdf, MARGIN, y, ["Feature", "Implementation"], features, [150, 365])


def scaling_page(pdf: Pdf, rows: list[dict[str, Any]]) -> None:
    pdf.add_page()
    y = page_title(pdf, "SIFT1M QPS Scaling")
    if not rows:
        y = paragraph(
            pdf,
            MARGIN,
            y,
            "No SIFT1M scaling result file was found. Run report/README.md commands "
            "to produce report/results/sift1m_scaling_results.json.",
            width=92,
        )
        return

    y = paragraph(
        pdf,
        MARGIN,
        y,
        "The current local scaling input comes from scaling_results.json. It measures "
        "coordinator search throughput while varying active reader count from 1 to 5. "
        "The expected Figure 10b-style artifact is a QPS-vs-reader-count curve.",
        width=92,
    )
    y += 12

    chart_qps(pdf, MARGIN, y, 500, 220, rows)
    y += 250
    chart_latency(pdf, MARGIN, y, 500, 220, rows)

    y += 250
    summary_rows = []
    base_qps = rows[0]["qps"] if rows and rows[0]["qps"] else 0.0
    for r in rows:
        speedup = r["qps"] / base_qps if base_qps else 0.0
        efficiency = speedup / r["readers"] if r["readers"] else 0.0
        summary_rows.append(
            [
                str(r["readers"]),
                f'{r["qps"]:.2f}',
                f'{r["p50_ms"]:.2f}',
                f'{r["p95_ms"]:.2f}',
                f'{r["p99_ms"]:.2f}',
                str(r["errors"]),
                f"{efficiency * 100:.1f}%",
            ]
        )
    table(
        pdf,
        MARGIN,
        y,
        ["Readers", "QPS", "p50", "p95", "p99", "Errors", "Efficiency"],
        summary_rows,
        [55, 70, 70, 70, 70, 55, 90],
        font_size=8,
    )


def comparison_page(
    pdf: Pdf,
    datasets: list[dict[str, Any]],
    benchmarks: dict[str, Any],
) -> None:
    scaling = benchmarks.get("scaling", {})
    qps_metrics = benchmarks.get("qps", {})
    recall_metrics = benchmarks.get("recall", {})
    load_metrics = benchmarks.get("load", {})
    pdf.add_page()
    y = page_title(pdf, "Cross-Dataset Evaluation Matrix")
    y = paragraph(
        pdf,
        MARGIN,
        y,
        "The report separates dataset readiness from benchmark readiness. A dataset can "
        "be available locally even if a full scaling or recall run has not been executed "
        "yet. This keeps the comparison honest and reproducible.",
        width=92,
    )
    y += 14

    rows = []
    for d in datasets:
        scale_rows = scaling.get(d["name"], [])
        qps_row = qps_metrics.get(d["name"], {})
        recall_row = recall_metrics.get(d["name"], {})
        load_row = load_metrics.get(d["name"], {})
        if scale_rows:
            best = max(scale_rows, key=lambda item: item["qps"])
            best_qps = f'{best["qps"]:.2f} @ {best["readers"]} readers'
            p95 = f'{best["p95_ms"]:.2f} ms'
            status = "measured"
        elif qps_row:
            best_qps = f'{qps_row.get("qps", 0.0):.2f}'
            p95 = f'{qps_row.get("p95_ms", 0.0):.2f} ms'
            status = "qps measured"
        else:
            best_qps = "pending"
            p95 = "pending"
            status = "dataset available" if d["available"] else "missing data"
        recall = recall_summary(recall_row)
        load_rate = (
            f'{load_row.get("vectors_per_second", 0.0):.0f}/s'
            if load_row
            else "pending"
        )
        rows.append(
            [
                d["name"],
                "yes" if d["available"] else "no",
                best_qps,
                p95,
                recall,
                load_rate,
                status,
            ]
        )
    y = table(
        pdf,
        MARGIN,
        y,
        ["Dataset", "Data", "Best QPS", "p95", "Recall", "Load", "Status"],
        rows,
        [68, 45, 95, 70, 80, 70, 95],
    )

    y += 26
    chart_coverage(pdf, MARGIN, y, 500, 210, datasets, scaling)

    y += 245
    pdf.text(MARGIN, y, "Interpretation", 13, bold=True, color=BLUE)
    y += 18
    bullets = [
        "SIFT10K validates correctness quickly and should report Recall@10/Recall@100.",
        "SIFT1M is the primary performance dataset and currently has local QPS scaling data.",
        "Deep1M is listed for paper replication but requires a local 96D HDF5 file and loader validation.",
        "The current SIFT1M run is not near-linear; QPS decreases as readers increase, which suggests local resource contention or benchmark setup effects that should be investigated before claiming positive scaling.",
    ]
    for item in bullets:
        y = bullet(pdf, MARGIN + 10, y, item)


def methodology_page(pdf: Pdf, metrics: dict[str, Any]) -> None:
    pdf.add_page()
    y = page_title(pdf, "Methodology And Reproduction")
    y = paragraph(
        pdf,
        MARGIN,
        y,
        "The report is generated from local files, not hard-coded benchmark claims. "
        "Run the commands below to refresh data, then rerun the report scripts.",
        width=92,
    )
    y += 16

    commands = [
        ("Collect metrics", "python report/collect_metrics.py"),
        ("Generate PDF", "python report/generate_report.py"),
        (
            "SIFT1M scaling",
            "python -m vecscaledb.bench.scaling_eval --dataset data/sift1m/sift1m.hdf5 --coordinator-url http://127.0.0.1:8000 --reader-counts 1 2 3 4 5 --duration 60 --concurrency 32 --query-limit 1000 --output report/results/sift1m_scaling_results.json --auto-scale",
        ),
        (
            "SIFT10K recall",
            "python -m vecscaledb.bench.recall_eval --host 127.0.0.1 --port 8000 --dataset data/sift1m/sift1m.hdf5",
        ),
    ]
    for title, command in commands:
        pdf.text(MARGIN, y, title, 10, bold=True)
        y += 14
        y = paragraph(pdf, MARGIN + 12, y, command, width=84, size=8, color=DARK_GRAY)
        y += 8

    pdf.text(MARGIN, y, "Metrics used", 13, bold=True, color=BLUE)
    y += 18
    for item in [
        "QPS: completed search requests divided by elapsed seconds.",
        "Latency p50/p95/p99: percentile search latency in milliseconds.",
        "Errors: failed or incomplete search requests in the benchmark window.",
        "Scaling efficiency: observed speedup divided by reader-count increase.",
        "Recall@k: overlap between returned top-k IDs and ground-truth nearest neighbors.",
    ]:
        y = bullet(pdf, MARGIN + 10, y, item)

    y += 10
    pdf.text(MARGIN, y, "Caveats", 13, bold=True, color=BLUE)
    y += 18
    for item in [
        "Milvus published values are not embedded numerically in this repo; the comparison reproduces the metric shape and dataset setup.",
        "Deep1M is marked pending until the local dataset is supplied.",
        "A single laptop Docker run can show resource contention, so final claims should be made from controlled repeated runs.",
    ]:
        y = bullet(pdf, MARGIN + 10, y, item)


def page_title(pdf: Pdf, title: str) -> float:
    y = 50
    pdf.text(MARGIN, y, title, 18, bold=True, color=BLUE)
    pdf.line(MARGIN, y + 10, PAGE_W - MARGIN, y + 10, LIGHT_GRAY, 1)
    return y + 32


def paragraph(
    pdf: Pdf,
    x: float,
    y: float,
    text: str,
    width: int = 80,
    size: float = 10,
    color: tuple[float, float, float] = BLACK,
) -> float:
    for line in wrap(text, width=width):
        pdf.text(x, y, line, size=size, color=color)
        y += size + 4
    return y


def bullet(
    pdf: Pdf,
    x: float,
    y: float,
    text: str,
    size: float = 9,
    color: tuple[float, float, float] = BLACK,
) -> float:
    pdf.text(x, y, "-", size=size, color=color)
    return paragraph(pdf, x + 12, y, text, width=84, size=size, color=color)


def key_value_table(
    pdf: Pdf,
    x: float,
    y: float,
    rows: list[tuple[str, str]],
    key_w: float,
    val_w: float,
) -> float:
    for key, value in rows:
        pdf.rect(x, y - 10, key_w, 22, fill=(0.94, 0.96, 0.98), stroke=LIGHT_GRAY)
        pdf.rect(x + key_w, y - 10, val_w, 22, fill=(1, 1, 1), stroke=LIGHT_GRAY)
        pdf.text(x + 7, y + 4, key, 9, bold=True)
        pdf.text(x + key_w + 7, y + 4, value, 9)
        y += 22
    return y


def table(
    pdf: Pdf,
    x: float,
    y: float,
    headers: list[str],
    rows: list[list[Any]],
    widths: list[float],
    font_size: float = 8.5,
) -> float:
    row_h = 24
    total_w = sum(widths)
    pdf.rect(x, y - 12, total_w, row_h, fill=BLUE, stroke=BLUE)
    cx = x
    for header, w in zip(headers, widths):
        pdf.text(cx + 5, y + 3, str(header), font_size, bold=True, color=WHITE)
        cx += w
    y += row_h
    for idx, row in enumerate(rows):
        fill = (0.97, 0.98, 0.99) if idx % 2 == 0 else (1, 1, 1)
        cx = x
        max_lines = 1
        wrapped_cells: list[list[str]] = []
        for cell, w in zip(row, widths):
            lines = wrap(str(cell), width=max(8, int(w / 5.0))) or [""]
            wrapped_cells.append(lines)
            max_lines = max(max_lines, len(lines))
        actual_h = max(row_h, 12 + max_lines * (font_size + 3))
        pdf.rect(x, y - 12, total_w, actual_h, fill=fill, stroke=LIGHT_GRAY)
        for lines, w in zip(wrapped_cells, widths):
            ly = y + 2
            for line in lines:
                pdf.text(cx + 5, ly, line, font_size)
                ly += font_size + 3
            cx += w
        y += actual_h
    return y


def chart_dataset_sizes(
    pdf: Pdf,
    x: float,
    y: float,
    w: float,
    h: float,
    datasets: list[dict[str, Any]],
) -> None:
    title(pdf, x, y, "Dataset scale, log10(vectors)")
    chart_x, chart_y = x + 30, y + 35
    chart_w, chart_h = w - 45, h - 65
    axis(pdf, chart_x, chart_y, chart_w, chart_h)
    values = [math.log10(d["expected_vectors"]) for d in datasets]
    max_v = max(values) or 1
    bar_w = chart_w / (len(datasets) * 1.7)
    for i, (d, v) in enumerate(zip(datasets, values)):
        bx = chart_x + 18 + i * (chart_w / len(datasets))
        bh = chart_h * v / max_v
        pdf.rect(bx, chart_y + chart_h - bh, bar_w, bh, fill=COLORS[i], stroke=COLORS[i])
        pdf.text(bx - 4, chart_y + chart_h + 14, d["name"], 7)
        pdf.text(bx - 2, chart_y + chart_h - bh - 5, f'{d["expected_vectors"]:,}', 7)


def chart_dataset_dims(
    pdf: Pdf,
    x: float,
    y: float,
    w: float,
    h: float,
    datasets: list[dict[str, Any]],
) -> None:
    title(pdf, x, y, "Vector dimensions")
    chart_x, chart_y = x + 30, y + 35
    chart_w, chart_h = w - 45, h - 65
    axis(pdf, chart_x, chart_y, chart_w, chart_h)
    max_v = max(d["expected_dims"] for d in datasets)
    bar_w = chart_w / (len(datasets) * 1.7)
    for i, d in enumerate(datasets):
        bx = chart_x + 18 + i * (chart_w / len(datasets))
        bh = chart_h * d["expected_dims"] / max_v
        pdf.rect(bx, chart_y + chart_h - bh, bar_w, bh, fill=COLORS[i], stroke=COLORS[i])
        pdf.text(bx - 4, chart_y + chart_h + 14, d["name"], 7)
        pdf.text(bx + 4, chart_y + chart_h - bh - 5, str(d["expected_dims"]), 7)


def chart_qps(pdf: Pdf, x: float, y: float, w: float, h: float, rows: list[dict[str, Any]]) -> None:
    title(pdf, x, y, "QPS vs reader nodes")
    chart = chart_area(pdf, x, y, w, h)
    xs = [r["readers"] for r in rows]
    ys = [r["qps"] for r in rows]
    line_chart(pdf, chart, xs, ys, "Readers", "QPS", BLUE)


def chart_latency(
    pdf: Pdf,
    x: float,
    y: float,
    w: float,
    h: float,
    rows: list[dict[str, Any]],
) -> None:
    title(pdf, x, y, "Latency percentiles vs reader nodes")
    chart = chart_area(pdf, x, y, w, h)
    xs = [r["readers"] for r in rows]
    ymax = max(max(r["p50_ms"], r["p95_ms"], r["p99_ms"]) for r in rows)
    axis(pdf, chart[0], chart[1], chart[2], chart[3])
    for label, key, color, offset in [
        ("p50", "p50_ms", BLUE, 0),
        ("p95", "p95_ms", ORANGE, 48),
        ("p99", "p99_ms", GREEN, 96),
    ]:
        points = _points(chart, xs, [r[key] for r in rows], max(xs), ymax)
        pdf.polyline(points, color, 1.6)
        for px, py in points:
            pdf.circle(px, py, 2.5, color)
        pdf.rect(chart[0] + offset, chart[1] - 22, 9, 9, fill=color)
        pdf.text(chart[0] + offset + 14, chart[1] - 13, label, 8)
    label_axes(pdf, chart, "Readers", "ms")


def chart_coverage(
    pdf: Pdf,
    x: float,
    y: float,
    w: float,
    h: float,
    datasets: list[dict[str, Any]],
    scaling: dict[str, list[dict[str, Any]]],
) -> None:
    title(pdf, x, y, "Dataset and benchmark coverage")
    chart_x, chart_y = x + 40, y + 38
    chart_w, chart_h = w - 70, h - 70
    labels = [d["name"] for d in datasets]
    categories = ["Data", "Scaling"]
    cell_w = chart_w / len(labels)
    cell_h = chart_h / len(categories)
    for i, d in enumerate(datasets):
        for j, cat in enumerate(categories):
            available = d["available"] if cat == "Data" else bool(scaling.get(d["name"]))
            color = GREEN if available else RED
            cx = chart_x + i * cell_w
            cy = chart_y + j * cell_h
            pdf.rect(cx, cy, cell_w - 8, cell_h - 8, fill=color, stroke=WHITE)
            pdf.text(cx + 12, cy + cell_h / 2, "yes" if available else "no", 10, bold=True, color=WHITE)
        pdf.text(chart_x + i * cell_w + 5, chart_y + chart_h + 12, labels[i], 8)
    for j, cat in enumerate(categories):
        pdf.text(chart_x - 38, chart_y + j * cell_h + cell_h / 2, cat, 8, bold=True)


def recall_summary(row: dict[str, Any]) -> str:
    if not row:
        return "pending"
    if row.get("skipped"):
        return "skipped"
    parts = []
    for key in sorted(row):
        if key.startswith("recall_at_"):
            label = "R@" + key.rsplit("_", 1)[-1]
            parts.append(f"{label}={row[key]:.3f}")
    return ", ".join(parts) if parts else "pending"


def flow_chart(pdf: Pdf, x: float, y: float, boxes: list[tuple[str, str]]) -> None:
    box_w, box_h = 155, 58
    gap_x, gap_y = 25, 28
    positions = []
    for i, (name, desc) in enumerate(boxes):
        row = i // 3
        col = i % 3
        bx = x + col * (box_w + gap_x)
        by = y + row * (box_h + gap_y)
        positions.append((bx, by))
        pdf.rect(bx, by, box_w, box_h, fill=(0.95, 0.97, 1.0), stroke=BLUE, width=1)
        pdf.text(bx + 8, by + 18, name, 11, bold=True, color=BLUE)
        paragraph(pdf, bx + 8, by + 34, desc, width=24, size=8, color=DARK_GRAY)
    for i in range(len(positions) - 1):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        if i == 2:
            continue
        pdf.line(x1 + box_w, y1 + box_h / 2, x2, y2 + box_h / 2, DARK_GRAY, 1)


def chart_area(pdf: Pdf, x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    chart = (x + 50, y + 36, w - 80, h - 72)
    axis(pdf, *chart)
    return chart


def axis(pdf: Pdf, x: float, y: float, w: float, h: float) -> None:
    pdf.line(x, y, x, y + h, LIGHT_GRAY, 1)
    pdf.line(x, y + h, x + w, y + h, LIGHT_GRAY, 1)


def line_chart(
    pdf: Pdf,
    chart: tuple[float, float, float, float],
    xs: list[int],
    ys: list[float],
    xlabel: str,
    ylabel: str,
    color: tuple[float, float, float],
) -> None:
    if not xs or not ys:
        return
    points = _points(chart, xs, ys, max(xs), max(ys) or 1.0)
    pdf.polyline(points, color, 2)
    for x, y in points:
        pdf.circle(x, y, 3, color)
    for xval, (px, py) in zip(xs, points):
        pdf.text(px - 3, chart[1] + chart[3] + 14, str(xval), 8)
        pdf.text(px - 10, py - 8, f"{ys[xs.index(xval)]:.1f}", 7)
    label_axes(pdf, chart, xlabel, ylabel)


def _points(
    chart: tuple[float, float, float, float],
    xs: list[int],
    ys: list[float],
    xmax: float,
    ymax: float,
) -> list[tuple[float, float]]:
    x, y, w, h = chart
    xmin = min(xs)
    xrange = max(xmax - xmin, 1)
    ymax = max(ymax, 1.0)
    points = []
    for xv, yv in zip(xs, ys):
        px = x + ((xv - xmin) / xrange) * w
        py = y + h - (yv / ymax) * h
        points.append((px, py))
    return points


def label_axes(pdf: Pdf, chart: tuple[float, float, float, float], xlabel: str, ylabel: str) -> None:
    x, y, w, h = chart
    pdf.text(x + w / 2 - 20, y + h + 32, xlabel, 8)
    pdf.text(x - 36, y + h / 2, ylabel, 8)


def title(pdf: Pdf, x: float, y: float, value: str) -> None:
    pdf.text(x, y, value, 11, bold=True, color=BLUE)


def _esc(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("\n", " ")
    )


if __name__ == "__main__":
    main()
