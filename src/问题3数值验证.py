"""问题三长时程数值验证：网格、内部步长、积分器与守恒残差。"""
from pathlib import Path
import sys
import time
from project_paths import DATA_ROOT, OUTPUT_ROOT

import numpy as np

ROOT = Path(__file__).resolve().parent
SUPPORT_ROOT = ROOT
sys.path.insert(0, str(ROOT))

from drying_core import build_fixed_radius_rhs, sample_radial_field, solve_fixed_radius
from 问题3 import FullProcessAmbient, INPUT, model_config, interpolate_crossing


RESULTS = OUTPUT_ROOT / "q3"
END_TIME_S = 60.0 * 3600.0
OUTPUT_TIMES = np.arange(0.0, END_TIME_S + 60.0, 60.0)
RADII_CM = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
CHECK_HOURS = np.array([6.0, 12.0, 24.0, 36.0, 48.0, 54.0])
CHECK_INDICES = (CHECK_HOURS * 60.0).astype(int)


def solve_case(label, n_intervals, method, max_step_s):
    started = time.perf_counter()
    ambient = FullProcessAmbient(INPUT)
    solution = solve_fixed_radius(
        model_config(), ambient, END_TIME_S, OUTPUT_TIMES,
        n_intervals=n_intervals, method=method, max_step_s=max_step_s,
        rtol=1.0e-9, atol=1.0e-11,
    )
    completion_s, _ = interpolate_crossing(solution.time_s, solution.moisture.max(axis=0))
    _, sampled_c = sample_radial_field(solution, RADII_CM / 100.0)
    elapsed = time.perf_counter() - started
    record = {
        "label": label,
        "n": n_intervals,
        "method": method,
        "max_step_s": max_step_s,
        "completion_h": completion_s / 3600.0,
        "sampled_c": sampled_c[CHECK_INDICES],
        "elapsed_s": elapsed,
        "solution": solution,
    }
    print(
        f"{label}: td={record['completion_h']:.10f} h, "
        f"nfev={solution.nfev}, elapsed={elapsed:.1f} s",
        flush=True,
    )
    return record


def sci(value):
    """表格单元用：自带数学模式。"""
    mantissa, exponent = f"{value:.3e}".split("e")
    return "$" + mantissa + r"\times10^{" + str(int(exponent)) + "}$"


def sci_plain(value):
    """宏定义用：不含数学模式，正文再以 $...$ 包裹。"""
    mantissa, exponent = f"{value:.3e}".split("e")
    return mantissa + r"\times10^{" + str(int(exponent)) + "}"


def balance_check(result, early_hours=(1.0, 3.0, 6.0), late_hours=(24.0, 54.0)):
    """守恒检查。

    预热阶段表面净热流非零，用相对残差；进入恒温阶段后表面温度与热风
    温度相等、净热流本身趋于零，此时相对残差失去意义（分母趋零），
    改报绝对残差并给出它与预热阶段热流的量级之比。
    """
    n_intervals = result.radius_m.size - 1
    _radius, _faces, volumes, rhs = build_fixed_radius_rhs(
        model_config(), FullProcessAmbient(INPUT), n_intervals
    )
    n_nodes = n_intervals + 1

    def one(hour):
        index = int(hour * 60.0)
        temperature_k = result.temperature_k[:, index]
        moisture = result.moisture[:, index]
        rate = rhs(result.time_s[index], np.concatenate((temperature_k, moisture)))
        props = model_config().properties
        rho = props.density(moisture, temperature_k)
        cp = props.heat_capacity(moisture, temperature_k)
        heat_storage = float(np.sum(volumes * rho * cp * rate[:n_nodes]))
        mass_storage = float(np.sum(volumes * rate[n_nodes:]))
        ambient_t_c, ambient_c = FullProcessAmbient(INPUT)(result.time_s[index])
        cfg = model_config()
        heat_boundary = (
            cfg.radius_m * cfg.heat_transfer_coefficient
            * ((float(np.asarray(ambient_t_c)) + 273.15) - temperature_k[-1])
        )
        mass_boundary = (
            cfg.radius_m * cfg.mass_transfer_coefficient
            * (float(np.asarray(ambient_c)) - moisture[-1])
        )
        return heat_storage, heat_boundary, mass_storage, mass_boundary

    early = [one(h) for h in early_hours]
    late = [one(h) for h in late_hours]
    heat_relative = max(
        abs(hs - hb) / max(abs(hb), 1.0e-30) for hs, hb, _ms, _mb in early
    )
    mass_relative = max(
        abs(ms - mb) / max(abs(mb), 1.0e-30)
        for _hs, _hb, ms, mb in early + late
    )
    heat_absolute_late = max(abs(hs - hb) for hs, hb, _ms, _mb in late)
    heat_reference = max(abs(hb) for _hs, hb, _ms, _mb in early)
    return heat_relative, mass_relative, heat_absolute_late, heat_reference


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    cases = {}
    specifications = [
        ("bdf_n800_s5", "BDF，800区间，5 s", 800, "BDF", 5.0),
        ("bdf_n200_s5", "BDF，200区间，5 s", 200, "BDF", 5.0),
        ("bdf_n400_s5", "BDF，400区间，5 s", 400, "BDF", 5.0),
        ("bdf_n1600_s5", "BDF，1600区间，5 s", 1600, "BDF", 5.0),
        ("bdf_n800_s20", "BDF，800区间，20 s", 800, "BDF", 20.0),
        ("bdf_n800_s10", "BDF，800区间，10 s", 800, "BDF", 10.0),
        ("radau_n800_s5", "Radau，800区间，5 s", 800, "Radau", 5.0),
    ]
    for key, label, n, method, max_step in specifications:
        cases[key] = solve_case(label, n, method, max_step)

    baseline = cases["bdf_n800_s5"]
    reference = cases["bdf_n1600_s5"]
    comparisons = []
    for key in ("bdf_n200_s5", "bdf_n400_s5", "bdf_n800_s5"):
        case = cases[key]
        comparisons.append((
            case["label"] + " vs. 1600区间",
            abs(case["completion_h"] - reference["completion_h"]) * 60.0,
            np.max(np.abs(case["sampled_c"] - reference["sampled_c"])),
        ))
    for key in ("bdf_n800_s20", "bdf_n800_s10"):
        case = cases[key]
        comparisons.append((
            case["label"] + " vs. 5 s",
            abs(case["completion_h"] - baseline["completion_h"]) * 60.0,
            np.max(np.abs(case["sampled_c"] - baseline["sampled_c"])),
        ))
    radau = cases["radau_n800_s5"]
    comparisons.append((
        "Radau vs. BDF（800区间，5 s）",
        abs(radau["completion_h"] - baseline["completion_h"]) * 60.0,
        np.max(np.abs(radau["sampled_c"] - baseline["sampled_c"])),
    ))

    table_lines = [
        f"{label} & {dt:.4f} & {sci(dc)} \\\\"
        for label, dt, dc in comparisons
    ]
    (RESULTS / "q3_validation_rows.tex").write_text(
        "\n".join(table_lines) + "\n\\bottomrule\n", encoding="utf-8"
    )
    csv_lines = ["comparison,completion_difference_min,max_moisture_difference"]
    csv_lines.extend(f'"{label}",{dt:.10f},{dc:.12e}' for label, dt, dc in comparisons)
    (RESULTS / "q3_validation.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8-sig")

    heat_relative, mass_relative, heat_absolute_late, heat_reference = balance_check(
        baseline["solution"]
    )

    grid800 = comparisons[2]
    step10 = comparisons[4]
    method_row = comparisons[-1]
    summary = (
        f"baseline_completion_h={baseline['completion_h']:.10f}\n"
        f"reference_completion_h={reference['completion_h']:.10f}\n"
        f"grid_800_completion_difference_min={grid800[1]:.10f}\n"
        f"grid_800_max_moisture_difference={grid800[2]:.12e}\n"
        f"step_10_completion_difference_min={step10[1]:.10f}\n"
        f"step_10_max_moisture_difference={step10[2]:.12e}\n"
        f"radau_completion_difference_min={method_row[1]:.10f}\n"
        f"radau_max_moisture_difference={method_row[2]:.12e}\n"
        f"heat_balance_relative_early={heat_relative:.12e}\n"
        f"heat_balance_absolute_late={heat_absolute_late:.12e}\n"
        f"heat_flux_reference={heat_reference:.6e}\n"
        f"mass_balance_relative={mass_relative:.12e}\n"
    )
    (RESULTS / "q3_validation_summary.txt").write_text(summary, encoding="utf-8")
    (RESULTS / "q3_validation_summary.tex").write_text(
        "\\newcommand{\\QThreeGridTime}{" + f"{grid800[1]:.4f}" + "}\n"
        "\\newcommand{\\QThreeGridC}{" + sci_plain(grid800[2]) + "}\n"
        "\\newcommand{\\QThreeStepTime}{" + f"{step10[1]:.4f}" + "}\n"
        "\\newcommand{\\QThreeStepC}{" + sci_plain(step10[2]) + "}\n"
        "\\newcommand{\\QThreeRadauTime}{" + f"{method_row[1]:.4f}" + "}\n"
        "\\newcommand{\\QThreeRadauC}{" + sci_plain(method_row[2]) + "}\n"
        "\\newcommand{\\QThreeHeatResidual}{" + sci_plain(heat_relative) + "}\n"
        "\\newcommand{\\QThreeHeatAbsLate}{" + sci_plain(heat_absolute_late) + "}\n"
        "\\newcommand{\\QThreeHeatRef}{" + sci_plain(heat_reference) + "}\n"
        "\\newcommand{\\QThreeMassResidual}{" + sci_plain(mass_relative) + "}\n",
        encoding="utf-8",
    )
    print(summary, end="", flush=True)


if __name__ == "__main__":
    main()
