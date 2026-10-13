"""圆柱形药材热湿传递模型的共用数值核心。

本模块统一管理几何、物性、边界条件、守恒型径向有限体积离散和
刚性时间积分接口。后续各问只替换物性关系、计算时段或几何映射，
从而避免符号、单位和边界条件不一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix


Array = np.ndarray
PropertyFunction = Callable[[Array, Array], Array]
AmbientFunction = Callable[[float | Array], tuple[Array, Array]]


@dataclass(frozen=True)
class MaterialProperties:
    """材料物性函数；温度输入采用 K，水分浓度采用 kg/kg。"""

    density: PropertyFunction
    heat_capacity: PropertyFunction
    conductivity: PropertyFunction
    moisture_diffusivity: PropertyFunction


@dataclass(frozen=True)
class CylinderConfig:
    """固定半径圆柱的模型配置。"""

    radius_m: float
    heat_transfer_coefficient: float
    mass_transfer_coefficient: float
    initial_temperature_k: float
    initial_moisture: float
    properties: MaterialProperties


@dataclass
class SimulationResult:
    """数值解；数组形状均为 (空间节点数, 时间点数)。"""

    time_s: Array
    radius_m: Array
    temperature_k: Array
    moisture: Array
    method: str
    nfev: int
    njev: int
    nlu: int

    @property
    def temperature_c(self) -> Array:
        return self.temperature_k - 273.15


def constant_property(value: float) -> PropertyFunction:
    """构造与温度、水分无关的常物性函数。"""

    def evaluate(moisture: Array, temperature_k: Array) -> Array:
        shape = np.broadcast_shapes(np.shape(moisture), np.shape(temperature_k))
        return np.full(shape, value, dtype=float)

    return evaluate


def radial_grid(radius_m: float, n_intervals: int) -> tuple[Array, Array, Array]:
    """返回节点、控制体积边界和去除公共 2*pi*L 因子的控制体积。"""

    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    if n_intervals < 4:
        raise ValueError("n_intervals must be at least 4")
    nodes = np.linspace(0.0, radius_m, n_intervals + 1)
    dr = radius_m / n_intervals
    faces = np.empty(n_intervals + 2)
    faces[0] = 0.0
    faces[1:-1] = (nodes[:-1] + nodes[1:]) / 2.0
    faces[-1] = radius_m
    volumes = 0.5 * (faces[1:] ** 2 - faces[:-1] ** 2)
    return nodes, faces, volumes


def harmonic_mean(left: Array, right: Array) -> Array:
    """正物性系数的调和平均，保证界面通量连续。"""

    tiny = np.finfo(float).tiny
    return 2.0 * left * right / np.maximum(left + right, tiny)


def jacobian_sparsity(n_nodes: int):
    """构造温度—水分耦合三点模板的稀疏结构。"""

    size = 2 * n_nodes
    pattern = lil_matrix((size, size), dtype=int)
    for equation_block in (0, 1):
        for i in range(n_nodes):
            row = equation_block * n_nodes + i
            for variable_block in (0, 1):
                for j in range(max(0, i - 1), min(n_nodes, i + 2)):
                    pattern[row, variable_block * n_nodes + j] = 1
    return pattern.tocsr()


def build_fixed_radius_rhs(
    config: CylinderConfig,
    ambient: AmbientFunction,
    n_intervals: int,
):
    """构造固定圆柱的一维轴对称守恒型半离散方程。"""

    radius, faces, volumes = radial_grid(config.radius_m, n_intervals)
    dr = config.radius_m / n_intervals
    n_nodes = n_intervals + 1

    def rhs(time_s: float, state: Array) -> Array:
        temperature_k = state[:n_nodes]
        moisture = state[n_nodes:]
        ambient_temperature_c, ambient_moisture = ambient(float(time_s))
        ambient_temperature_k = float(np.asarray(ambient_temperature_c)) + 273.15
        ambient_moisture_value = float(np.asarray(ambient_moisture))

        rho = config.properties.density(moisture, temperature_k)
        cp = config.properties.heat_capacity(moisture, temperature_k)
        conductivity = config.properties.conductivity(moisture, temperature_k)
        diffusivity = config.properties.moisture_diffusivity(moisture, temperature_k)

        if np.any(rho <= 0) or np.any(cp <= 0):
            raise FloatingPointError("density and heat capacity must stay positive")
        if np.any(conductivity <= 0) or np.any(diffusivity <= 0):
            raise FloatingPointError("transport coefficients must stay positive")

        conductivity_face = harmonic_mean(conductivity[:-1], conductivity[1:])
        diffusivity_face = harmonic_mean(diffusivity[:-1], diffusivity[1:])

        heat_flux_coordinate = np.zeros(n_nodes + 1)
        mass_flux_coordinate = np.zeros(n_nodes + 1)
        heat_flux_coordinate[1:-1] = (
            faces[1:-1]
            * conductivity_face
            * np.diff(temperature_k)
            / dr
        )
        mass_flux_coordinate[1:-1] = (
            faces[1:-1]
            * diffusivity_face
            * np.diff(moisture)
            / dr
        )

        # Gamma*du/dr = h*(u_inf-u_s)，与外法向通量形式完全等价。
        heat_flux_coordinate[-1] = (
            config.radius_m
            * config.heat_transfer_coefficient
            * (ambient_temperature_k - temperature_k[-1])
        )
        mass_flux_coordinate[-1] = (
            config.radius_m
            * config.mass_transfer_coefficient
            * (ambient_moisture_value - moisture[-1])
        )

        heat_divergence = np.diff(heat_flux_coordinate) / volumes
        mass_divergence = np.diff(mass_flux_coordinate) / volumes
        temperature_rate = heat_divergence / (rho * cp)
        moisture_rate = mass_divergence
        return np.concatenate((temperature_rate, moisture_rate))

    return radius, faces, volumes, rhs


def solve_fixed_radius(
    config: CylinderConfig,
    ambient: AmbientFunction,
    end_time_s: float,
    output_times_s: Array,
    *,
    n_intervals: int = 200,
    method: Literal["BDF", "Radau"] = "BDF",
    max_step_s: float = 5.0,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
) -> SimulationResult:
    """求解固定半径圆柱模型。"""

    output_times_s = np.asarray(output_times_s, dtype=float)
    if output_times_s.ndim != 1 or output_times_s.size == 0:
        raise ValueError("output_times_s must be a non-empty one-dimensional array")
    if output_times_s[0] < 0 or output_times_s[-1] > end_time_s:
        raise ValueError("output times must lie in [0, end_time_s]")
    if np.any(np.diff(output_times_s) <= 0):
        raise ValueError("output times must be strictly increasing")

    radius, _faces, _volumes, rhs = build_fixed_radius_rhs(
        config, ambient, n_intervals
    )
    n_nodes = n_intervals + 1
    initial_state = np.concatenate(
        (
            np.full(n_nodes, config.initial_temperature_k),
            np.full(n_nodes, config.initial_moisture),
        )
    )
    solution = solve_ivp(
        rhs,
        (0.0, float(end_time_s)),
        initial_state,
        method=method,
        t_eval=output_times_s,
        max_step=max_step_s,
        rtol=rtol,
        atol=atol,
        jac_sparsity=jacobian_sparsity(n_nodes),
    )
    if not solution.success:
        raise RuntimeError(f"time integration failed: {solution.message}")
    return SimulationResult(
        time_s=solution.t,
        radius_m=radius,
        temperature_k=solution.y[:n_nodes],
        moisture=solution.y[n_nodes:],
        method=method,
        nfev=solution.nfev,
        njev=solution.njev,
        nlu=solution.nlu,
    )


def sample_radial_field(result: SimulationResult, radii_m: Array) -> tuple[Array, Array]:
    """在线性径向坐标上抽取温度与水分结果。"""

    radii_m = np.asarray(radii_m, dtype=float)
    if np.any(radii_m < 0) or np.any(radii_m > result.radius_m[-1]):
        raise ValueError("requested radii lie outside the cylinder")
    temperature = np.vstack(
        [np.interp(radii_m, result.radius_m, column) for column in result.temperature_c.T]
    )
    moisture = np.vstack(
        [np.interp(radii_m, result.radius_m, column) for column in result.moisture.T]
    )
    return temperature, moisture


def conservative_balance_residuals(
    config: CylinderConfig,
    ambient: AmbientFunction,
    result: SimulationResult,
    sample_indices: Array,
) -> tuple[float, float]:
    """检查半离散方程的整体热量和水分通量平衡。"""

    n_intervals = result.radius_m.size - 1
    _radius, _faces, volumes, rhs = build_fixed_radius_rhs(
        config, ambient, n_intervals
    )
    n_nodes = n_intervals + 1
    heat_residuals: list[float] = []
    mass_residuals: list[float] = []
    for index in np.asarray(sample_indices, dtype=int):
        time_s = float(result.time_s[index])
        temperature_k = result.temperature_k[:, index]
        moisture = result.moisture[:, index]
        state = np.concatenate((temperature_k, moisture))
        rate = rhs(time_s, state)
        rho = config.properties.density(moisture, temperature_k)
        cp = config.properties.heat_capacity(moisture, temperature_k)
        ambient_temperature_c, ambient_moisture = ambient(time_s)
        heat_storage = float(np.sum(volumes * rho * cp * rate[:n_nodes]))
        mass_storage = float(np.sum(volumes * rate[n_nodes:]))
        heat_boundary = (
            config.radius_m
            * config.heat_transfer_coefficient
            * (float(np.asarray(ambient_temperature_c)) + 273.15 - temperature_k[-1])
        )
        mass_boundary = (
            config.radius_m
            * config.mass_transfer_coefficient
            * (float(np.asarray(ambient_moisture)) - moisture[-1])
        )
        heat_scale = max(abs(heat_boundary), 1.0e-30)
        mass_scale = max(abs(mass_boundary), 1.0e-30)
        heat_residuals.append(abs(heat_storage - heat_boundary) / heat_scale)
        mass_residuals.append(abs(mass_storage - mass_boundary) / mass_scale)
    return max(heat_residuals), max(mass_residuals)
