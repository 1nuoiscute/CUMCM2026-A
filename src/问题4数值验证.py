"""问题四独立数值验证：网格、最大步长和移动区域水分守恒。

本脚本只调用《问题4.py》中的求解函数，不运行其 main()，不覆盖 result4.xlsx、
原始图件或基准汇总。全部输出使用 q4_numerical_validation 前缀。
"""
from __future__ import annotations

import csv
import importlib.util
import sys
import time
from project_paths import DATA_ROOT, OUTPUT_ROOT
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_simpson


ROOT = Path(__file__).resolve().parent
RESULTS = OUTPUT_ROOT / "q4"
FIGURES = RESULTS
MODEL_PATH = ROOT / "问题4.py"
ATTACHMENT_DIR = DATA_ROOT
ATTACHMENT1 = ATTACHMENT_DIR / "附件1.xlsx"

END_TIME_S = 56.0 * 3600.0
OUTPUT_TIMES_S = np.arange(0.0, END_TIME_S + 30.0, 30.0)
CHECK_HOURS = (6.0, 24.0, 48.0)
RTOL = 1.0e-7
ATOL = 1.0e-9


def load_model():
    spec = importlib.util.spec_from_file_location("q4_model_for_validation", MODEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载问题4.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_baseline_parameters() -> tuple[float, float, float]:
    values: dict[str, float] = {}
    path = RESULTS / "q4_summary.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            try:
                values[key] = float(value)
            except ValueError:
                pass
    return values["shrinkage_a"], values["shrinkage_b"], values["completion_time_h"]


@dataclass
class Case:
    key: str
    group: str
    n: int
    max_step_s: float
    result: object
    completion_s: float
    completion_radius_cm: float
    completion_surface_c: float
    elapsed_s: float


def solve_case(model, ambient, radius_function, key: str, group: str,
               n: int, max_step_s: float) -> Case:
    started = time.perf_counter()
    result = model.solve_moving(
        ambient,
        radius_function,
        END_TIME_S,
        OUTPUT_TIMES_S,
        n_intervals=n,
        max_step_s=max_step_s,
        rtol=RTOL,
        atol=ATOL,
    )
    completion_s, _ = model.completion_time(result)
    profile = model.field_at_time(result.time_s, result.moisture, completion_s)
    radius_m = model.scalar_at_time(result.time_s, result.radius_m, completion_s)
    elapsed_s = time.perf_counter() - started
    case = Case(
        key=key,
        group=group,
        n=n,
        max_step_s=max_step_s,
        result=result,
        completion_s=completion_s,
        completion_radius_cm=radius_m * 100.0,
        completion_surface_c=float(profile[-1]),
        elapsed_s=elapsed_s,
    )
    print(
        f"{key}: t4={completion_s/3600.0:.10f} h, "
        f"R={case.completion_radius_cm:.9f} cm, "
        f"Cs={case.completion_surface_c:.9f}, "
        f"nfev={result.nfev}, elapsed={elapsed_s:.1f} s",
        flush=True,
    )
    return case


def profile_error(model, case: Case, reference: Case, hour: float) -> float:
    target_s = hour * 3600.0
    field_case = model.field_at_time(case.result.time_s, case.result.moisture, target_s)
    field_ref = model.field_at_time(reference.result.time_s, reference.result.moisture, target_s)
    case_on_ref = np.interp(reference.result.xi, case.result.xi, field_case)
    return float(np.max(np.abs(case_on_ref - field_ref)))


def comparison_metrics(model, case: Case, reference: Case) -> dict[str, float]:
    errors = {f"field_error_{int(hour)}h": profile_error(model, case, reference, hour)
              for hour in CHECK_HOURS}
    return {
        "completion_difference_min": abs(case.completion_s-reference.completion_s) / 60.0,
        "radius_difference_cm": abs(case.completion_radius_cm-reference.completion_radius_cm),
        "surface_difference": abs(case.completion_surface_c-reference.completion_surface_c),
        **errors,
        "field_error_max": max(errors.values()),
    }


def moving_balance(model, ambient, case: Case) -> tuple[float, float, float]:
    """返回累计残差、瞬时代数残差和终点累计排湿量。

    移动区域的面积权重写入材料坐标：Cbar=2*int_0^1 xi*C dxi。
    表面排湿满足 dCbar/dt=-2*Jout/R。累计残差以初始储量C0归一化。
    """
    result = case.result
    mean_c = model.mean_field(result, result.moisture)
    c_inf = np.array([ambient(float(t))[1] for t in result.time_s])
    jout = model.HM * (result.moisture[-1] - c_inf)
    discharge_rate = 2.0 * jout / result.radius_m
    cumulative_out = cumulative_simpson(discharge_rate, x=result.time_s, initial=0.0)
    cumulative_residual = np.abs((model.C0 - mean_c) - cumulative_out) / model.C0
    completion_mask = result.time_s <= case.completion_s
    max_cumulative = float(np.max(cumulative_residual[completion_mask]))

    xi, faces, volumes, dxi = model.fixed_xi_grid(case.n)
    weights = volumes / volumes.sum()
    instant = []
    indices = np.linspace(0, np.flatnonzero(completion_mask)[-1], 121, dtype=int)
    for index in indices:
        C = result.moisture[:, index]
        T = result.temperature_k[:, index]
        R = result.radius_m[index]
        Cinf = ambient(float(result.time_s[index]))[1]
        D = model.diffusivity_q4(C, T)
        Df = model.harmonic_mean(D[:-1], D[1:])
        mass_coordinate = np.zeros(C.size + 1)
        mass_coordinate[1:-1] = faces[1:-1] * Df * np.diff(C) / dxi
        mass_coordinate[-1] = R * model.HM * (Cinf - C[-1])
        rate = np.diff(mass_coordinate) / (volumes * R * R)
        storage_rate = float(weights @ rate)
        boundary_rate = float(2.0 * mass_coordinate[-1] / (R * R))
        instant.append(abs(storage_rate-boundary_rate) / max(abs(boundary_rate), 1.0e-30))
    return max_cumulative, float(max(instant)), float(cumulative_out[completion_mask][-1])


def sci_plain(value: float, digits: int = 3) -> str:
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return mantissa + r"\times10^{" + str(int(exponent)) + "}"


def write_outputs(model, cases: dict[str, Case], metrics: dict[str, dict[str, float]],
                  cumulative_residual: float, instant_residual: float,
                  cumulative_out: float, baseline_target_h: float) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    header = [
        "group", "case", "n_intervals", "max_step_s", "completion_h",
        "completion_radius_cm", "completion_surface_C", "completion_difference_min",
        "radius_difference_cm", "surface_difference", "field_error_6h",
        "field_error_24h", "field_error_48h", "field_error_max", "elapsed_s", "nfev",
    ]
    with (RESULTS / "q4_numerical_validation.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        for key in ("grid_n150", "grid_n300", "grid_n600", "step_s60", "step_s30", "step_s15"):
            case = cases[key]
            writer.writerow({
                "group": case.group,
                "case": key,
                "n_intervals": case.n,
                "max_step_s": f"{case.max_step_s:.1f}",
                "completion_h": f"{case.completion_s/3600.0:.10f}",
                "completion_radius_cm": f"{case.completion_radius_cm:.10f}",
                "completion_surface_C": f"{case.completion_surface_c:.10f}",
                **{name: f"{value:.12e}" for name, value in metrics[key].items()},
                "elapsed_s": f"{case.elapsed_s:.3f}",
                "nfev": case.result.nfev,
            })

    grid300 = metrics["grid_n300"]
    step30 = metrics["step_s30"]
    baseline = cases["grid_n300"]
    summary = (
        f"baseline_reported_completion_h={baseline_target_h:.10f}\n"
        f"validation_n300_s30_completion_h={baseline.completion_s/3600.0:.10f}\n"
        f"validation_baseline_difference_s={abs(baseline.completion_s/3600.0-baseline_target_h)*3600.0:.12e}\n"
        f"grid_n300_vs_n600_completion_difference_min={grid300['completion_difference_min']:.12e}\n"
        f"grid_n300_vs_n600_radius_difference_cm={grid300['radius_difference_cm']:.12e}\n"
        f"grid_n300_vs_n600_surface_difference={grid300['surface_difference']:.12e}\n"
        f"grid_n300_vs_n600_field_error_max={grid300['field_error_max']:.12e}\n"
        f"step_30_vs_15_completion_difference_min={step30['completion_difference_min']:.12e}\n"
        f"step_30_vs_15_radius_difference_cm={step30['radius_difference_cm']:.12e}\n"
        f"step_30_vs_15_surface_difference={step30['surface_difference']:.12e}\n"
        f"step_30_vs_15_field_error_max={step30['field_error_max']:.12e}\n"
        f"moving_cumulative_balance_residual={cumulative_residual:.12e}\n"
        f"moving_instant_balance_residual={instant_residual:.12e}\n"
        f"cumulative_surface_discharge_at_last_precompletion={cumulative_out:.12e}\n"
        f"rtol={RTOL:.1e}\n"
        f"atol={ATOL:.1e}\n"
    )
    (RESULTS / "q4_numerical_validation_summary.txt").write_text(summary, encoding="utf-8")
    (RESULTS / "q4_numerical_validation_summary.tex").write_text(
        "\\newcommand{\\QFourGridTime}{" + sci_plain(grid300["completion_difference_min"]) + "}\n"
        "\\newcommand{\\QFourGridRadius}{" + sci_plain(grid300["radius_difference_cm"]) + "}\n"
        "\\newcommand{\\QFourGridSurface}{" + sci_plain(grid300["surface_difference"]) + "}\n"
        "\\newcommand{\\QFourGridField}{" + sci_plain(grid300["field_error_max"]) + "}\n"
        "\\newcommand{\\QFourStepTime}{" + sci_plain(step30["completion_difference_min"]) + "}\n"
        "\\newcommand{\\QFourStepRadius}{" + sci_plain(step30["radius_difference_cm"]) + "}\n"
        "\\newcommand{\\QFourStepSurface}{" + sci_plain(step30["surface_difference"]) + "}\n"
        "\\newcommand{\\QFourStepField}{" + sci_plain(step30["field_error_max"]) + "}\n"
        "\\newcommand{\\QFourMovingCumResidual}{" + sci_plain(cumulative_residual) + "}\n"
        "\\newcommand{\\QFourMovingInstantResidual}{" + sci_plain(instant_residual) + "}\n",
        encoding="utf-8",
    )


def configure_plot() -> None:
    available = {font.name for font in mpl.font_manager.fontManager.ttflist}
    font = "Microsoft YaHei" if "Microsoft YaHei" in available else "SimHei"
    mpl.rcParams.update({
        "font.sans-serif": [font, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 320,
    })


def make_convergence_figure(cases: dict[str, Case], metrics: dict[str, dict[str, float]]) -> None:
    configure_plot()
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.72), constrained_layout=True)

    n = np.array([150, 300])
    grid_error = np.array([
        metrics["grid_n150"]["completion_difference_min"],
        metrics["grid_n300"]["completion_difference_min"],
    ])
    axes[0].semilogy(n, grid_error, color="#355C7D", marker="o", markersize=4.3)
    axes[0].set(xticks=n, xlabel="径向区间数 $N$", ylabel="烘干完成时间差 / min")
    axes[0].set_title("(a) 网格收敛（以 $N=600$ 为参考）", fontsize=9.4, pad=4)

    steps = np.array([60.0, 30.0])
    step_error = 60.0 * np.array([
        metrics["step_s60"]["completion_difference_min"],
        metrics["step_s30"]["completion_difference_min"],
    ])
    axes[1].semilogy(steps, step_error, color="#2A9D8F", marker="o", markersize=4.3)
    axes[1].set(xticks=steps, xlabel="最大步长上限 / s", ylabel="烘干完成时间差 / s")
    axes[1].invert_xaxis()
    axes[1].set_title("(b) 最大步长敏感性（以15 s为参考）", fontsize=9.4, pad=4)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, which="major", color="#D9E1E8", linewidth=0.55, alpha=0.8)
        ax.grid(True, which="minor", color="#EDF1F4", linewidth=0.35, alpha=0.65)
        ax.set_axisbelow(True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    base = FIGURES / "q4_numerical_convergence"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    model = load_model()
    a, b, reported_completion_h = read_baseline_parameters()
    ambient = model.FullProcessAmbient(ATTACHMENT1)
    radius_function = lambda _t, cbar: model.R0 * model.shrinkage_ratio(cbar/model.C0, a, b)

    cases: dict[str, Case] = {}
    specs = [
        ("grid_n150", "grid", 150, 30.0),
        ("grid_n300", "grid", 300, 30.0),
        ("grid_n600", "grid", 600, 30.0),
        ("step_s60", "max_step", 300, 60.0),
        ("step_s15", "max_step", 300, 15.0),
    ]
    for key, group, n, step in specs:
        cases[key] = solve_case(model, ambient, radius_function, key, group, n, step)
    cases["step_s30"] = Case(**{**cases["grid_n300"].__dict__, "key": "step_s30", "group": "max_step"})

    grid_ref = cases["grid_n600"]
    step_ref = cases["step_s15"]
    metrics: dict[str, dict[str, float]] = {}
    for key in ("grid_n150", "grid_n300", "grid_n600"):
        metrics[key] = comparison_metrics(model, cases[key], grid_ref)
    for key in ("step_s60", "step_s30", "step_s15"):
        metrics[key] = comparison_metrics(model, cases[key], step_ref)

    cumulative_residual, instant_residual, cumulative_out = moving_balance(
        model, ambient, cases["grid_n300"]
    )
    write_outputs(
        model, cases, metrics, cumulative_residual, instant_residual,
        cumulative_out, reported_completion_h
    )
    make_convergence_figure(cases, metrics)

    baseline_difference_s = abs(cases["grid_n300"].completion_s/3600.0-reported_completion_h)*3600.0
    if baseline_difference_s > 0.05:
        raise AssertionError(f"验证基准与原结果相差 {baseline_difference_s:.6g} s")
    if metrics["grid_n300"]["completion_difference_min"] > 3.0:
        raise AssertionError("N=300相对N=600的终点差超过3 min")
    if metrics["step_s30"]["completion_difference_min"] > 0.01:
        raise AssertionError("30 s相对15 s的终点差超过0.01 min")
    if cumulative_residual > 1.0e-4 or instant_residual > 1.0e-12:
        raise AssertionError("移动区域水分守恒残差超过预设容限")
    print("问题四数值验证完成，断言全部通过。", flush=True)


if __name__ == "__main__":
    main()
