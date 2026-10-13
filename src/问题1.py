"""问题一：圆柱形药材预热阶段的温度和水分浓度计算。

运行方式：
    python 问题1.py

程序读取 data/附件1.xlsx，生成 result1.xlsx、问题一论文表格、
验证数据和全部问题一图片。所有随机过程均固定种子；本问的确定性计算
不依赖随机数。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from project_paths import DATA_ROOT, OUTPUT_ROOT

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

QUESTION_DIR = Path(__file__).resolve().parent
SUPPORT_ROOT = QUESTION_DIR
CONTEST_ROOT = QUESTION_DIR.parent
sys.path.insert(0, str(QUESTION_DIR))

from drying_core import (
    CylinderConfig,
    MaterialProperties,
    conservative_balance_residuals,
    constant_property,
    radial_grid,
    sample_radial_field,
    solve_fixed_radius,
)


RANDOM_SEED = 20260910
np.random.seed(RANDOM_SEED)

INPUT_FILE = DATA_ROOT / "附件1.xlsx"
DATA_DIR = OUTPUT_ROOT / "q1"
FIGURE_DIR = OUTPUT_ROOT / "q1"
RESULT_FILE = FIGURE_DIR / "result1.xlsx"

TABLE_TIMES_S = np.array([100, 300, 600, 900, 1200, 1500, 1800], dtype=float)
TABLE_RADII_CM = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
OUTPUT_RADII_CM = np.round(np.arange(0.0, 2.0 + 0.05, 0.1), 1)


@dataclass(frozen=True)
class AmbientHistory:
    time_s: np.ndarray
    temperature_c: np.ndarray
    moisture: np.ndarray

    def __call__(self, query_time_s):
        query = np.asarray(query_time_s, dtype=float)
        temperature = np.interp(
            query, self.time_s, self.temperature_c,
            left=self.temperature_c[0], right=self.temperature_c[-1]
        )
        moisture = np.interp(
            query, self.time_s, self.moisture,
            left=self.moisture[0], right=self.moisture[-1]
        )
        return temperature, moisture


def read_ambient_history(path: Path) -> AmbientHistory:
    if not path.exists():
        raise FileNotFoundError(f"未找到附件1：{path}")
    worksheet = load_workbook(path, data_only=True, read_only=True).active
    rows = [row for row in worksheet.iter_rows(min_row=2, values_only=True) if row[0] is not None]
    data = np.asarray(rows, dtype=float)
    if data.shape[1] < 3:
        raise ValueError("附件1应至少包含时间、温度和水分浓度三列")
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("附件1时间列必须严格递增")
    return AmbientHistory(data[:, 0], data[:, 1], data[:, 2])


def q1_config() -> CylinderConfig:
    properties = MaterialProperties(
        density=constant_property(820.0),
        heat_capacity=constant_property(2600.0),
        conductivity=constant_property(0.36),
        moisture_diffusivity=lambda moisture, temperature_k: (
            7.0e-9 * np.exp(-0.89 / np.maximum(moisture, 1.0e-12))
        ),
    )
    return CylinderConfig(
        radius_m=0.02,
        heat_transfer_coefficient=25.0,
        mass_transfer_coefficient=8.0e-7,
        initial_temperature_k=28.0 + 273.15,
        initial_moisture=2.55,
        properties=properties,
    )


def configure_chinese_font() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"]
    available = {font.name for font in mpl.font_manager.fontManager.ttflist}
    selected = next((name for name in candidates if name in available), "DejaVu Sans")
    mpl.rcParams.update({
        "font.sans-serif": [selected, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def write_result_workbook(times_s, radii_cm, temperature, moisture) -> None:
    workbook = Workbook()
    workbook.remove(workbook.active)
    header_fill = PatternFill("solid", fgColor="D9EAF7")
    for sheet_name, field in (("温度", temperature), ("水分浓度", moisture)):
        worksheet = workbook.create_sheet(sheet_name)
        worksheet.cell(1, 1, "时间\\到药材中心的距离")
        for column, radius_cm in enumerate(radii_cm, start=2):
            worksheet.cell(1, column, float(radius_cm))
        for row_index, time_s in enumerate(times_s, start=2):
            worksheet.cell(row_index, 1, int(round(float(time_s))))
            for column, value in enumerate(field[row_index - 2], start=2):
                cell = worksheet.cell(row_index, column, round(float(value), 4))
                cell.number_format = "0.0000"
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")
        worksheet.freeze_panes = "B2"
        worksheet.column_dimensions["A"].width = 25
        for column in range(2, len(radii_cm) + 2):
            worksheet.column_dimensions[worksheet.cell(1, column).column_letter].width = 11
    workbook.save(RESULT_FILE)


def write_table_products(table_temperature, table_moisture) -> None:
    header = ["time_s"] + [f"r_{radius:g}_cm" for radius in TABLE_RADII_CM]
    write_csv(
        DATA_DIR / "q1_temperature_table.csv", header,
        [[int(time_s), *[f"{value:.4f}" for value in row]]
         for time_s, row in zip(TABLE_TIMES_S, table_temperature)]
    )
    write_csv(
        DATA_DIR / "q1_moisture_table.csv", header,
        [[int(time_s), *[f"{value:.4f}" for value in row]]
         for time_s, row in zip(TABLE_TIMES_S, table_moisture)]
    )
    for file_name, values in (
        ("q1_temperature_rows.tex", table_temperature),
        ("q1_moisture_rows.tex", table_moisture),
    ):
        lines = []
        for time_s, row in zip(TABLE_TIMES_S, values):
            cells = " & ".join(f"{value:.4f}" for value in row)
            lines.append(f"{int(time_s)} & {cells} \\\\")
        (DATA_DIR / file_name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_products(config, ambient, solution) -> None:
    """保存体积平均量和量纲分析指标，供正文结果分析直接调用。"""

    _nodes, _faces, volumes = radial_grid(
        config.radius_m, solution.radius_m.size - 1
    )
    weights = volumes / np.sum(volumes)
    rows = []
    for time_s in TABLE_TIMES_S.astype(int):
        ambient_temperature, ambient_moisture = ambient(float(time_s))
        rows.append([
            time_s,
            f"{solution.temperature_c[0, time_s]:.8f}",
            f"{solution.temperature_c[-1, time_s]:.8f}",
            f"{np.dot(weights, solution.temperature_c[:, time_s]):.8f}",
            f"{float(np.asarray(ambient_temperature)):.8f}",
            f"{solution.moisture[0, time_s]:.8f}",
            f"{solution.moisture[-1, time_s]:.8f}",
            f"{np.dot(weights, solution.moisture[:, time_s]):.8f}",
            f"{float(np.asarray(ambient_moisture)):.8f}",
        ])
    write_csv(
        DATA_DIR / "q1_summary_history.csv",
        [
            "time_s", "center_temperature_C", "surface_temperature_C",
            "volume_mean_temperature_C", "ambient_temperature_C",
            "center_moisture", "surface_moisture", "volume_mean_moisture",
            "ambient_moisture",
        ],
        rows,
    )

    initial_diffusivity = 7.0e-9 * np.exp(-0.89 / config.initial_moisture)
    thermal_diffusivity = 0.36 / (820.0 * 2600.0)
    characteristic_length = config.radius_m / 2.0
    dimensionless_rows = [
        ["aspect_ratio_L_over_R", f"{0.25 / config.radius_m:.10f}"],
        ["end_face_area_fraction", f"{config.radius_m / (0.25 + config.radius_m):.10f}"],
        ["thermal_Biot", f"{config.heat_transfer_coefficient * characteristic_length / 0.36:.10f}"],
        ["initial_mass_Biot", f"{config.mass_transfer_coefficient * characteristic_length / initial_diffusivity:.10f}"],
        ["thermal_Fourier_at_1800s", f"{thermal_diffusivity * 1800.0 / config.radius_m**2:.10f}"],
        ["initial_mass_Fourier_at_1800s", f"{initial_diffusivity * 1800.0 / config.radius_m**2:.10f}"],
    ]
    write_csv(
        DATA_DIR / "q1_dimensionless_numbers.csv",
        ["metric", "value"],
        dimensionless_rows,
    )


def compare_at_targets(reference, candidate):
    radii_m = TABLE_RADII_CM / 100.0
    ref_t, ref_c = sample_radial_field(reference, radii_m)
    cur_t, cur_c = sample_radial_field(candidate, radii_m)
    return float(np.max(np.abs(cur_t - ref_t))), float(np.max(np.abs(cur_c - ref_c)))


def run_validation(config, ambient, final_solution):
    validation_times = TABLE_TIMES_S
    radii_m = TABLE_RADII_CM / 100.0
    fine_reference = solve_fixed_radius(
        config, ambient, 1800.0, validation_times,
        n_intervals=1600, method="BDF", max_step_s=5.0,
        rtol=1.0e-9, atol=1.0e-11,
    )
    reference_temperature, reference_moisture = sample_radial_field(
        fine_reference, radii_m
    )

    grid_rows = []
    for n_intervals in (50, 100, 200, 400):
        solution = solve_fixed_radius(
            config, ambient, 1800.0, validation_times,
            n_intervals=n_intervals, method="BDF", max_step_s=5.0,
            rtol=1.0e-9, atol=1.0e-11,
        )
        temperature, moisture = sample_radial_field(solution, radii_m)
        grid_rows.append([
            n_intervals,
            f"{2.0 / n_intervals:.6f}",
            f"{np.max(np.abs(temperature - reference_temperature)):.10e}",
            f"{np.max(np.abs(moisture - reference_moisture)):.10e}",
            solution.nfev,
        ])
    final_temperature_all, final_moisture_all = sample_radial_field(
        final_solution, radii_m
    )
    target_indices = TABLE_TIMES_S.astype(int)
    final_temperature = final_temperature_all[target_indices]
    final_moisture = final_moisture_all[target_indices]
    grid_rows.append([
        800, f"{2.0 / 800:.6f}",
        f"{np.max(np.abs(final_temperature - reference_temperature)):.10e}",
        f"{np.max(np.abs(final_moisture - reference_moisture)):.10e}",
        final_solution.nfev,
    ])
    grid_rows.append([1600, f"{2.0 / 1600:.6f}", "0.0", "0.0", fine_reference.nfev])
    write_csv(
        DATA_DIR / "q1_grid_convergence.csv",
        ["radial_intervals", "delta_r_cm", "max_abs_temperature_C", "max_abs_moisture", "nfev"],
        grid_rows,
    )

    time_solutions = {}
    for max_step in (10.0, 5.0, 2.5):
        time_solutions[max_step] = solve_fixed_radius(
            config, ambient, 1800.0, validation_times,
            n_intervals=200, method="BDF", max_step_s=max_step,
            rtol=1.0e-9, atol=1.0e-11,
        )
    time_reference = time_solutions[2.5]
    validation_rows = []
    for max_step in (10.0, 5.0, 2.5):
        temperature_error, moisture_error = compare_at_targets(
            time_reference, time_solutions[max_step]
        )
        validation_rows.append([
            f"BDF_max_step_{max_step:g}s", 200, "BDF", max_step,
            f"{temperature_error:.10e}", f"{moisture_error:.10e}",
            time_solutions[max_step].nfev,
        ])

    bdf_100 = solve_fixed_radius(
        config, ambient, 1800.0, validation_times,
        n_intervals=100, method="BDF", max_step_s=5.0,
        rtol=1.0e-9, atol=1.0e-11,
    )
    radau_100 = solve_fixed_radius(
        config, ambient, 1800.0, validation_times,
        n_intervals=100, method="Radau", max_step_s=5.0,
        rtol=1.0e-9, atol=1.0e-11,
    )
    temperature_error, moisture_error = compare_at_targets(bdf_100, radau_100)
    validation_rows.append([
        "Radau_vs_BDF", 100, "Radau/BDF", 5.0,
        f"{temperature_error:.10e}", f"{moisture_error:.10e}",
        radau_100.nfev,
    ])
    write_csv(
        DATA_DIR / "q1_time_method_validation.csv",
        ["test", "radial_intervals", "method", "max_step_s", "max_abs_temperature_C", "max_abs_moisture", "nfev"],
        validation_rows,
    )

    sample_indices = np.array([100, 300, 600, 900, 1200, 1500, 1800])
    heat_balance, mass_balance = conservative_balance_residuals(
        config, ambient, final_solution, sample_indices
    )
    diagnostics = [
        ["temperature_min_C", f"{final_solution.temperature_c.min():.12g}"],
        ["temperature_max_C", f"{final_solution.temperature_c.max():.12g}"],
        ["moisture_min_kg_per_kg", f"{final_solution.moisture.min():.12g}"],
        ["moisture_max_kg_per_kg", f"{final_solution.moisture.max():.12g}"],
        ["max_relative_heat_balance_residual", f"{heat_balance:.12e}"],
        ["max_relative_mass_balance_residual", f"{mass_balance:.12e}"],
        ["final_solver_nfev", final_solution.nfev],
        ["final_solver_njev", final_solution.njev],
        ["final_solver_nlu", final_solution.nlu],
    ]
    write_csv(DATA_DIR / "q1_diagnostics.csv", ["metric", "value"], diagnostics)
    return grid_rows, validation_rows


def save_figure(fig, file_name: str) -> None:
    fig.savefig(FIGURE_DIR / file_name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_physical_picture() -> None:
    """绘制圆柱中部截面及热量内传、水分外迁的物理图景。"""

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), constrained_layout=True)
    ax = axes[0]
    ax.add_patch(Rectangle((0.7, 0.8), 7.2, 1.55, facecolor="#edc77f",
                           edgecolor="#5b4636", lw=1.8))
    ax.plot([4.3, 4.3], [0.55, 2.62], "--", color="#64748b", lw=1.3)
    ax.annotate("轴向长度 25 cm", xy=(0.7, 0.38), xytext=(7.9, 0.38),
                arrowprops=dict(arrowstyle="<->", color="#334155"),
                ha="center", va="center")
    ax.text(4.3, 2.75, "取中部横截面建立一维径向模型", ha="center", fontsize=10)
    ax.text(4.3, 1.58, "长径比 12.5，忽略端部效应", ha="center", color="#5b4636")
    ax.set_xlim(0, 8.7)
    ax.set_ylim(0, 3.15)
    ax.set_aspect("equal")
    ax.axis("off")

    ax = axes[1]
    ax.add_patch(Circle((0, 0), 2.0, facecolor="#f6d9a5",
                        edgecolor="#5b4636", lw=2.0))
    ax.add_patch(FancyArrowPatch((0, 0), (2.0, 0), arrowstyle="-|>",
                                 mutation_scale=13, color="#334155", lw=1.5))
    ax.text(1.0, 0.14, "$r$", ha="center", color="#334155", fontsize=12)
    ax.text(0, -0.22, "$r=0$", ha="center")
    ax.text(2.03, -0.22, "$r=R$", ha="left")

    # 红色箭头从热风指向药材，表示热量由外向内。
    for angle in np.deg2rad([0, 90, 180, 270]):
        outside = 2.65 * np.array([np.cos(angle), np.sin(angle)])
        surface = 2.02 * np.array([np.cos(angle), np.sin(angle)])
        ax.add_patch(FancyArrowPatch(outside, surface, arrowstyle="-|>",
                                     mutation_scale=12, color="#c2410c", lw=1.7))
    # 蓝色箭头从药材表面指向热风，表示水分由内向外。
    for angle in np.deg2rad([45, 135, 225, 315]):
        surface = 1.95 * np.array([np.cos(angle), np.sin(angle)])
        outside = 2.62 * np.array([np.cos(angle), np.sin(angle)])
        ax.add_patch(FancyArrowPatch(surface, outside, arrowstyle="-|>",
                                     mutation_scale=12, color="#0369a1", lw=1.7))
    ax.text(0, 2.92, r"热风边界：$T_\infty(t)>T_s$", ha="center", color="#9a3412")
    ax.text(-2.85, -2.55, r"热量 $q_T$：由外向内", ha="left", color="#c2410c")
    ax.text(-2.85, -2.88, r"水分 $j_C$：由内向外", ha="left", color="#0369a1")
    ax.set_xlim(-3.05, 3.05)
    ax.set_ylim(-3.05, 3.15)
    ax.set_aspect("equal")
    ax.axis("off")
    save_figure(fig, "q1_physical_picture.png")


def plot_boundary_conditions() -> None:
    """绘制中心对称条件、表面 Robin 条件和两个通量方向。"""

    fig, ax = plt.subplots(figsize=(10.5, 4.15), constrained_layout=True)
    ax.add_patch(Rectangle((2.2, 1.25), 5.2, 2.15, facecolor="#f7e3b8",
                           edgecolor="#8b5e3c", lw=1.8))
    ax.add_patch(Rectangle((7.4, 0.65), 2.5, 3.35, facecolor="#e8f2f8",
                           edgecolor="none", alpha=0.95))
    ax.plot([2.2, 2.2], [1.0, 3.65], "--", color="#475569", lw=1.5)
    ax.plot([7.4, 7.4], [0.75, 3.9], color="#5b4636", lw=2.2)
    ax.text(4.8, 3.08, r"药材内部：$0\leq r\leq R$", ha="center", fontsize=11)
    ax.text(4.8, 2.34, r"$T(r,0)=T_0,\quad C(r,0)=C_0$", ha="center", fontsize=11)
    ax.text(1.15, 3.62, "中心对称边界", ha="center", fontsize=10, color="#334155")
    ax.text(1.15, 2.95, r"$r=0$", ha="center", fontsize=11)
    ax.text(1.15, 2.30, r"$\partial T/\partial r=0$", ha="center")
    ax.text(1.15, 1.80, r"$\partial C/\partial r=0$", ha="center")
    ax.text(8.65, 3.72, r"热风：$T_\infty(t),\ C_\infty(t)$", ha="center", fontsize=10)
    ax.text(7.42, 0.42, r"$r=R$", ha="center", fontsize=11)

    ax.add_patch(FancyArrowPatch((9.35, 3.0), (7.48, 3.0), arrowstyle="-|>",
                                 mutation_scale=15, color="#c2410c", lw=2.2))
    ax.text(8.42, 3.18, r"$q_T=h(T_s-T_\infty)<0$", ha="center",
            color="#9a3412", fontsize=10)
    ax.add_patch(FancyArrowPatch((7.48, 1.55), (9.35, 1.55), arrowstyle="-|>",
                                 mutation_scale=15, color="#0369a1", lw=2.2))
    ax.text(8.42, 1.75, r"$j_C=h_m(C_s-C_\infty)>0$", ha="center",
            color="#075985", fontsize=10)
    ax.text(8.65, 1.05, "外向为正：负热流表示热量内传，正水分流表示水分外迁",
            ha="center", fontsize=9, color="#334155")
    ax.annotate("径向坐标 $r$", xy=(7.15, 0.78), xytext=(2.45, 0.78),
                arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.4),
                ha="left", va="center")
    ax.set_xlim(0, 10.1)
    ax.set_ylim(0.1, 4.15)
    ax.axis("off")
    save_figure(fig, "q1_boundary_conditions.png")


def plot_computation_flowchart() -> None:
    """绘制从附件和题设参数到结果与验证的计算流程。"""

    fig, ax = plt.subplots(figsize=(11.2, 4.7), constrained_layout=True)

    def box(x, y, w, h, label, color):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=color, edgecolor="#475569", lw=1.25
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.2)
        return patch

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                     mutation_scale=13, color="#475569", lw=1.35))

    box(0.15, 3.05, 1.85, 0.85, "附件1\n60 s边界数据", "#dbeafe")
    box(0.15, 1.35, 1.85, 0.85, "题设参数\n$R, h, h_m, D(C)$", "#fef3c7")
    box(2.55, 3.05, 1.95, 0.85,
        "分段线性插值\n" + r"$T_\infty(t), C_\infty(t)$", "#e0f2fe")
    box(2.55, 1.35, 1.95, 0.85,
        "物性函数与初值\n" + r"$T_0, C_0, \rho, c_p, k$", "#fef3c7")
    box(5.05, 2.20, 2.15, 1.05, "守恒型径向有限体积\n中心零通量、表面Robin条件\n界面物性调和平均", "#dcfce7")
    box(7.75, 2.20, 1.75, 1.05, "稀疏刚性方程组\nBDF隐式积分", "#ede9fe")
    box(10.05, 3.05, 1.85, 0.90, "统一结果数组\n每1 s、每0.1 cm", "#fee2e2")
    box(10.05, 1.15, 1.85, 1.10, "水分与温度表、Excel、图\n网格/步长/Radau/守恒验证", "#f1f5f9")

    arrow((2.0, 3.48), (2.55, 3.48))
    arrow((2.0, 1.78), (2.55, 1.78))
    arrow((4.50, 3.48), (5.05, 2.92))
    arrow((4.50, 1.78), (5.05, 2.48))
    arrow((7.20, 2.72), (7.75, 2.72))
    arrow((9.50, 2.72), (10.05, 3.38))
    arrow((10.98, 3.05), (10.98, 2.25))
    ax.text(6.12, 3.65, r"$\rho c_p\dot T=\nabla\!\cdot(k\nabla T)$",
            ha="center", fontsize=9, color="#166534")
    ax.text(6.12, 1.72, r"$\dot C=\nabla\!\cdot(D\nabla C)$",
            ha="center", fontsize=9, color="#166534")
    ax.set_xlim(0, 12.1)
    ax.set_ylim(0.6, 4.3)
    ax.axis("off")
    save_figure(fig, "q1_computation_flowchart.png")


def plot_boundary(ambient: AmbientHistory) -> None:
    mask = ambient.time_s <= 1800.0
    time_min = ambient.time_s[mask] / 60.0
    fig, axis_temperature = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    axis_moisture = axis_temperature.twinx()
    line1 = axis_temperature.plot(time_min, ambient.temperature_c[mask], "o-", ms=3.0,
                                  lw=1.6, color="#c2410c", label="烘房温度")
    line2 = axis_moisture.plot(time_min, ambient.moisture[mask], "s-", ms=2.8,
                              lw=1.5, color="#0369a1", label="环境水分浓度")
    axis_temperature.set_xlabel("时间 / min")
    axis_temperature.set_ylabel("烘房温度 / °C", color="#c2410c")
    axis_moisture.set_ylabel("环境水分浓度 / (kg/kg)", color="#0369a1")
    axis_temperature.tick_params(axis="y", colors="#c2410c")
    axis_moisture.tick_params(axis="y", colors="#0369a1")
    axis_temperature.legend(line1 + line2, [x.get_label() for x in line1 + line2], loc="upper left")
    save_figure(fig, "q1_boundary_history.png")


def plot_profiles(solution) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.15), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(TABLE_TIMES_S)))
    for index, time_s in enumerate(TABLE_TIMES_S):
        time_index = int(time_s)
        radius_cm = solution.radius_m * 100.0
        axes[0].plot(radius_cm, solution.temperature_c[:, time_index],
                     color=colors[index], label=f"{int(time_s)} s")
        axes[1].plot(radius_cm, solution.moisture[:, time_index],
                     color=colors[index], label=f"{int(time_s)} s")
    axes[0].set_xlabel("到药材中心的距离 / cm")
    axes[0].set_ylabel("温度 / °C")
    axes[0].set_title("径向温度剖面")
    axes[1].set_xlabel("到药材中心的距离 / cm")
    axes[1].set_ylabel("水分浓度 / (kg/kg)")
    axes[1].set_title("径向水分浓度剖面")
    axes[1].legend(ncol=1, fontsize=8, frameon=False, loc="best")
    save_figure(fig, "q1_radial_profiles.png")


def plot_contours(solution) -> None:
    time_min = solution.time_s / 60.0
    radius_cm = solution.radius_m * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.25), constrained_layout=True)
    image_temperature = axes[0].pcolormesh(
        time_min, radius_cm, solution.temperature_c, shading="auto", cmap="inferno"
    )
    image_moisture = axes[1].pcolormesh(
        time_min, radius_cm, solution.moisture, shading="auto", cmap="YlGnBu_r"
    )
    axes[0].set_title("温度时空分布")
    axes[1].set_title("水分浓度时空分布")
    for axis in axes:
        axis.set_xlabel("时间 / min")
        axis.set_ylabel("到药材中心的距离 / cm")
        axis.grid(False)
    fig.colorbar(image_temperature, ax=axes[0], label="温度 / °C")
    fig.colorbar(image_moisture, ax=axes[1], label="水分浓度 / (kg/kg)")
    save_figure(fig, "q1_spatiotemporal_contours.png")


def plot_center_surface_history(solution, ambient) -> None:
    ambient_temperature, ambient_moisture = ambient(solution.time_s)
    time_min = solution.time_s / 60.0
    fig, axes = plt.subplots(2, 1, figsize=(8.3, 6.2), sharex=True, constrained_layout=True)
    axes[0].plot(time_min, ambient_temperature, "--", color="#6b7280", label="烘房")
    axes[0].plot(time_min, solution.temperature_c[0], color="#1d4ed8", label="中心")
    axes[0].plot(time_min, solution.temperature_c[-1], color="#dc2626", label="表面")
    axes[0].set_ylabel("温度 / °C")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].set_title("中心—表面响应与烘房边界")
    axes[1].plot(time_min, ambient_moisture, "--", color="#6b7280", label="烘房")
    axes[1].plot(time_min, solution.moisture[0], color="#1d4ed8", label="中心")
    axes[1].plot(time_min, solution.moisture[-1], color="#059669", label="表面")
    axes[1].set_xlabel("时间 / min")
    axes[1].set_ylabel("水分浓度 / (kg/kg)")
    axes[1].legend(frameon=False, ncol=3)
    save_figure(fig, "q1_center_surface_history.png")


def plot_convergence(grid_rows) -> None:
    intervals = np.array([float(row[0]) for row in grid_rows[:-1]])
    temperature_error = np.array([float(row[2]) for row in grid_rows[:-1]])
    moisture_error = np.array([float(row[3]) for row in grid_rows[:-1]])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), constrained_layout=True)
    axes[0].loglog(intervals, temperature_error, "o-", color="#b91c1c")
    axes[1].loglog(intervals, moisture_error, "o-", color="#047857")
    axes[0].set_ylabel("最大温度绝对偏差 / °C")
    axes[1].set_ylabel("最大水分绝对偏差 / (kg/kg)")
    for axis in axes:
        axis.set_xlabel("径向区间数 $N$")
        axis.set_xticks(intervals)
        axis.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        axis.get_xaxis().set_minor_locator(mpl.ticker.NullLocator())
    axes[0].set_title("温度网格收敛")
    axes[1].set_title("水分网格收敛")
    save_figure(fig, "q1_grid_convergence.png")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    configure_chinese_font()
    ambient = read_ambient_history(INPUT_FILE)
    config = q1_config()

    output_times = np.arange(0.0, 1800.0 + 1.0, 1.0)
    final_solution = solve_fixed_radius(
        config,
        ambient,
        1800.0,
        output_times,
        n_intervals=800,
        method="BDF",
        max_step_s=5.0,
        rtol=1.0e-9,
        atol=1.0e-11,
    )

    output_temperature, output_moisture = sample_radial_field(
        final_solution, OUTPUT_RADII_CM / 100.0
    )
    # 模板从第1秒开始，不写入初始时刻。
    write_result_workbook(
        final_solution.time_s[1:], OUTPUT_RADII_CM,
        output_temperature[1:], output_moisture[1:]
    )

    table_indices = TABLE_TIMES_S.astype(int)
    table_temperature = output_temperature[table_indices][:, [0, 5, 10, 15, 20]]
    table_moisture = output_moisture[table_indices][:, [0, 5, 10, 15, 20]]
    write_table_products(table_temperature, table_moisture)
    write_summary_products(config, ambient, final_solution)

    grid_rows, _validation_rows = run_validation(config, ambient, final_solution)
    plot_boundary(ambient)
    plot_profiles(final_solution)

    print("问题一计算完成")
    print(f"主结果：{RESULT_FILE}")
    print(f"1800 s中心温度：{table_temperature[-1, 0]:.6f} °C")
    print(f"1800 s表面温度：{table_temperature[-1, -1]:.6f} °C")
    print(f"1800 s中心水分浓度：{table_moisture[-1, 0]:.6f} kg/kg")
    print(f"1800 s表面水分浓度：{table_moisture[-1, -1]:.6f} kg/kg")


if __name__ == "__main__":
    main()
