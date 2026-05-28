from __future__ import annotations

import pandas as pd

from src.dashboard.visuals import WhatIfScenario, what_if_simulation


def build_scenario(sales_growth_pct: float, seasonality_multiplier: float, inventory_capacity_multiplier: float, demand_shift_pct: float) -> WhatIfScenario:
    return WhatIfScenario(
        sales_growth_pct=sales_growth_pct,
        seasonality_multiplier=seasonality_multiplier,
        inventory_capacity_multiplier=inventory_capacity_multiplier,
        demand_shift_pct=demand_shift_pct,
    )


def run_what_if(frame: pd.DataFrame, sales_growth_pct: float, seasonality_multiplier: float, inventory_capacity_multiplier: float, demand_shift_pct: float) -> pd.DataFrame:
    scenario = build_scenario(sales_growth_pct, seasonality_multiplier, inventory_capacity_multiplier, demand_shift_pct)
    return what_if_simulation(frame, scenario)
