"""Retail inventory control with demand taken from the M5 (Walmart) dataset (real-data bonus problem).

One product on one store shelf, reviewed once a day:

    state   (units on hand at the start of the day, weekday); weekday 0 = Monday. Only states
            reachable from the start are kept (a full shelf on Thursday would need a Wednesday
            with no sales, which never happens in the data), because online methods must be
            able to visit every state
    action  order 0..capacity units; delivery is immediate, and only the units that fit on the
            shelf (stock + order <= capacity) are received and paid for
    demand  drawn from that weekday's empirical sales distribution in the M5 data
    cost    fixed_order * [any units received] + unit_cost * received
            + holding * units left overnight - price * units sold
            (negative daily profit; an unmet sale is lost and costs its missing revenue)

Default costs (modelling assumptions; M5 gives only sales and shelf prices): unit cost 70% of
the average shelf price ($1.65 -> $1.15), $1 per delivery, $0.05 per unit held overnight.
With these, the optimal policy reorders when stock falls to 1-3 units and fills up to 8-11,
with higher levels before and during the weekend.

Demand is independent from day to day given the weekday, so the exact model is known and every
policy can be scored exactly, as for GridWorld and Blackjack.

Data: M5 Forecasting competition (Makridakis Open Forecasting Center, University of Nicosia;
Walmart unit sales 2011-01-29 to 2016-05-22, https://www.kaggle.com/c/m5-forecasting-accuracy),
downloaded from Nixtla's public mirror (M5_URL). The extracted series ships with the package in
``envs/data/`` so the raw files (about 490 MB unzipped) are not needed.

Regenerate the extracted series (downloads about 50 MB):
    python -m ie692_rl.envs.inventory --extract
"""

import argparse
import csv
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit

from ie692_rl.envs.base import Environment, Kernel, TabularModel
from ie692_rl.types import FloatArray, IntArray

M5_URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "m5_raw"  # git-ignored
ITEM, STORE, DAYS = "FOODS_3_377", "WI_1", 730
SERIES_FILE = DATA_DIR / f"m5_{ITEM}_{STORE}.csv"
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


# ----------------------------------------------------------------------------- data


def download_m5(raw_dir: Path = RAW_DIR) -> Path:
    """Download and unzip the files needed from the M5 mirror (skipped if already there)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    needed = ["sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"]
    if all((raw_dir / name).exists() for name in needed):
        return raw_dir
    archive = raw_dir / "m5.zip"
    if not archive.exists():
        urllib.request.urlretrieve(M5_URL, archive)
    with zipfile.ZipFile(archive) as zipped:
        for name in needed:
            zipped.extract(name, raw_dir)
    return raw_dir


def extract_series(
    raw_dir: Path = RAW_DIR, item: str = ITEM, store: str = STORE, days: int = DAYS, out: Path = SERIES_FILE
) -> Path:
    """Write the last ``days`` days of one product-store series: date, weekday, units sold, price."""
    with open(raw_dir / "sales_train_evaluation.csv") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        first_day = header.index("d_1")
        row = next(
            r for r in reader if r[header.index("item_id")] == item and r[header.index("store_id")] == store
        )
    sales = [int(x) for x in row[first_day:]]
    with open(raw_dir / "calendar.csv") as handle:
        calendar = list(csv.DictReader(handle))[: len(sales)]
    prices = {}
    with open(raw_dir / "sell_prices.csv") as handle:
        for r in csv.DictReader(handle):
            if r["item_id"] == item and r["store_id"] == store:
                prices[r["wm_yr_wk"]] = float(r["sell_price"])
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "weekday", "units_sold", "sell_price"])
        for day, units in zip(calendar[-days:], sales[-days:], strict=True):
            writer.writerow([day["date"], day["weekday"], units, prices.get(day["wm_yr_wk"], "")])
    return out


def load_series(path: Path = SERIES_FILE) -> tuple[IntArray, IntArray, float]:
    """(units sold per day, weekday index per day with Monday = 0, average shelf price)."""
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    units = np.array([int(r["units_sold"]) for r in rows], dtype=np.int64)
    weekdays = np.array([WEEKDAYS.index(r["weekday"]) for r in rows], dtype=np.int64)
    price = float(np.mean([float(r["sell_price"]) for r in rows if r["sell_price"]]))
    return units, weekdays, price


# ----------------------------------------------------------------------------- problem


class RetailInventory(Environment):
    """Single-product inventory MDP with empirical, weekday-dependent demand (costs, gamma 0.9)."""

    def __init__(
        self,
        units: IntArray,
        weekdays: IntArray,
        price: float,
        capacity: int | None = None,
        unit_cost_fraction: float = 0.7,
        holding: float = 0.05,
        fixed_order: float = 1.0,
        gamma: float = 0.9,
        weekly: bool = True,
        start_stock: int = 0,
        start_weekday: int = 0,
    ) -> None:
        units, weekdays = np.asarray(units, dtype=np.int64), np.asarray(weekdays, dtype=np.int64)
        if len(units) == 0 or units.min() < 0 or len(units) != len(weekdays):
            raise ValueError("Need non-negative unit sales with one weekday per day")
        if not 0 < gamma < 1 or price <= 0:
            raise ValueError("Need 0 < gamma < 1 and a positive price")
        self.capacity = int(np.percentile(units, 99)) if capacity is None else int(capacity)
        self.price, self.unit_cost = float(price), unit_cost_fraction * float(price)
        self.holding, self.fixed_order, self.gamma = float(holding), float(fixed_order), float(gamma)
        self.weekly = weekly
        self.n_days = 7 if weekly else 1
        self.n_stock = self.capacity + 1
        self.n_actions = self.capacity + 1
        if not (0 <= start_stock <= self.capacity and 0 <= start_weekday < 7):
            raise ValueError("Need 0 <= start_stock <= capacity and 0 <= start_weekday < 7")
        # Demand distribution for each weekday (or one pooled distribution), over 0..max sold.
        groups = weekdays if weekly else np.zeros_like(weekdays)
        self.demand_pmf = np.zeros((self.n_days, int(units.max()) + 1))
        for day in range(self.n_days):
            counts = np.bincount(units[groups == day], minlength=units.max() + 1)
            if counts.sum() == 0:
                raise ValueError(f"No sales data for weekday {day}")
            self.demand_pmf[day] = counts / counts.sum()
        self._demand_cdf = np.cumsum(self.demand_pmf, axis=1)
        self._demand_cdf[:, -1] = 1.0  # guard against rounding: demand never exceeds the largest sale
        self._keep_reachable(start_weekday * self.n_stock + start_stock if weekly else start_stock)
        self.start = int(self.encode(start_stock, start_weekday if weekly else 0))
        # Largest one-step cost: pay for a full delivery, hold everything, sell nothing.
        self.cost_bound = self.fixed_order + self.unit_cost * self.capacity + self.holding * self.capacity
        self._kernel = Kernel(
            _step,
            _reset,
            (
                self.n_stock,
                self.n_days,
                self.capacity,
                self.price,
                self.unit_cost,
                self.holding,
                self.fixed_order,
                self._demand_cdf,
                self._cells,
                self._state_of_cell,
                self.start,
            ),
        )

    def _outcome(
        self, stock: IntArray, day: IntArray, order: IntArray, demand: IntArray
    ) -> tuple[FloatArray, IntArray, IntArray]:
        """Cost, units left and next day for given demands (elementwise; same rules as ``_step``)."""
        on_shelf = np.minimum(stock + order, self.capacity)
        received = on_shelf - stock
        sold = np.minimum(on_shelf, demand)
        left = on_shelf - sold
        cost = (
            self.fixed_order * (received > 0)
            + self.unit_cost * received
            + self.holding * left
            - self.price * sold
        )
        return cost, left, (day + 1) % self.n_days

    def _keep_reachable(self, start: int) -> None:
        """Number only the (stock, day) cells reachable from the start cell (cell = day * n_stock + stock)."""
        reached, frontier = {start}, [start]
        while frontier:
            cell = frontier.pop()
            stock, day = np.array([cell % self.n_stock]), np.array([cell // self.n_stock])
            demands = np.flatnonzero(self.demand_pmf[day[0]])
            for order in range(self.n_actions):
                _, left, next_day = self._outcome(stock, day, np.array([order]), demands)
                for nxt in {int(c) for c in next_day * self.n_stock + left} - reached:
                    reached.add(nxt)
                    frontier.append(nxt)
        self._cells = np.array(sorted(reached), dtype=np.int64)  # state index -> cell
        self._state_of_cell = np.full(self.n_stock * self.n_days, -1, dtype=np.int64)  # -1: unreachable
        self._state_of_cell[self._cells] = np.arange(len(self._cells))
        self.n_states = len(self._cells)

    @classmethod
    def from_m5(cls, path: Path | str = SERIES_FILE, **settings: Any) -> "RetailInventory":
        units, weekdays, price = load_series(Path(path))
        return cls(units, weekdays, price, **settings)

    @property
    def kernel(self) -> Kernel:
        return self._kernel

    def encode(self, stock: int | IntArray, day: int | IntArray) -> IntArray:
        """State index of (stock, day); -1 if that combination is unreachable."""
        result: IntArray = self._state_of_cell[np.asarray(day) * self.n_stock + np.asarray(stock)]
        return result

    def decode(self, states: int | IntArray) -> tuple[IntArray, IntArray]:
        """(stock, day)."""
        cells = self._cells[np.asarray(states)]
        return cells % self.n_stock, cells // self.n_stock

    def model(self) -> TabularModel:
        """Sparse model with one slot per possible demand value."""
        n_demand = self.demand_pmf.shape[1]
        demands = np.arange(n_demand)
        next_states = np.zeros((self.n_states, self.n_actions, n_demand), dtype=np.int64)
        probs = np.zeros((self.n_states, self.n_actions, n_demand))
        costs = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            stock, day = self.decode(np.array([s]))
            pmf = self.demand_pmf[day[0]]
            for a in range(self.n_actions):
                cost, left, next_day = self._outcome(stock, day, np.array([a]), demands)
                costs[s, a] = pmf @ cost
                reachable = pmf > 0  # unreachable successors only appear with probability 0
                next_states[s, a, reachable] = self.encode(left[reachable], next_day)
                probs[s, a] = pmf
        return TabularModel(next_states, probs, costs)

    def state_label(self, state: int) -> dict[str, int]:
        stock, day = self.decode(state)
        return {"stock": int(stock), "weekday": int(day)}


@njit(cache=True)
def _step(
    params: tuple[Any, ...], state: int, action: int, rng: np.random.Generator
) -> tuple[int, float, bool]:
    n_stock, n_days, capacity, price, unit_cost, holding, fixed_order, cdf, cells, state_of_cell, _ = params
    cell = cells[state]
    stock, day = cell % n_stock, cell // n_stock
    u = rng.random()
    demand = 0
    for k in range(cdf.shape[1]):
        if u > cdf[day, k]:
            demand += 1
    on_shelf = min(stock + action, capacity)
    received = on_shelf - stock
    sold = min(on_shelf, demand)
    left = on_shelf - sold
    cost = (fixed_order if received > 0 else 0.0) + unit_cost * received + holding * left - price * sold
    next_state: int = state_of_cell[((day + 1) % n_days) * n_stock + left]
    return next_state, cost, False


@njit(cache=True)
def _reset(params: tuple[Any, ...], rng: np.random.Generator) -> int:
    start: int = params[10]
    return start


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--extract", action="store_true", help=f"download M5 and rewrite {SERIES_FILE.name}")
    if parser.parse_args().extract:
        DATA_DIR.mkdir(exist_ok=True)
        print("wrote", extract_series(download_m5()))
    env = RetailInventory.from_m5()
    print(
        f"{ITEM} at {STORE}: capacity {env.capacity}, price ${env.price:.2f}, "
        f"{env.n_states} states x {env.n_actions} actions"
    )


if __name__ == "__main__":
    main()
