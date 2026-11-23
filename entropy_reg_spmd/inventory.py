"""Retail inventory control with demand taken from the M5 (Walmart) dataset.

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

Demand is independent from day to day given the weekday, so the exact transition model is
known and every policy can be scored exactly (exact_model, start_distribution), like the
team's GridWorld and blackjack. The interface matches theirs: n_states, n_actions, gamma,
step(states, actions, rng) -> (next_states, costs, done) and initial_states(count, rng).

Data: M5 Forecasting competition (Makridakis Open Forecasting Center, University of Nicosia;
Walmart unit sales 2011-01-29 to 2016-05-22, https://www.kaggle.com/c/m5-forecasting-accuracy),
downloaded from Nixtla's public mirror (M5_URL). The extracted series used here is stored in
data/ so the raw files (about 490 MB unzipped) are not needed to run the problem.

Regenerate the extracted series (downloads about 50 MB):
    python entropy_reg_spmd/inventory.py --extract
"""
import argparse
import csv
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

M5_URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "m5_raw"  # git-ignored
ITEM, STORE, DAYS = "FOODS_3_377", "WI_1", 730
SERIES_FILE = DATA_DIR / f"m5_{ITEM}_{STORE}.csv"
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


# ----------------------------------------------------------------------------- data

def download_m5(raw_dir=RAW_DIR):
    """Download and unzip the files needed from the M5 mirror (skipped if already there)."""
    raw_dir = Path(raw_dir)
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


def extract_series(raw_dir=RAW_DIR, item=ITEM, store=STORE, days=DAYS, out=SERIES_FILE):
    """Write the last `days` days of one product-store series: date, weekday, units sold, price."""
    raw_dir = Path(raw_dir)
    with open(raw_dir / "sales_train_evaluation.csv") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        first_day = header.index("d_1")  # this mirror has no leading "id" column
        row = next(r for r in reader if r[header.index("item_id")] == item and r[header.index("store_id")] == store)
    sales = [int(x) for x in row[first_day:]]
    calendar = list(csv.DictReader(open(raw_dir / "calendar.csv")))[:len(sales)]
    prices = {}
    with open(raw_dir / "sell_prices.csv") as handle:
        for r in csv.DictReader(handle):
            if r["item_id"] == item and r["store_id"] == store:
                prices[r["wm_yr_wk"]] = float(r["sell_price"])
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "weekday", "units_sold", "sell_price"])
        for day, units in zip(calendar[-days:], sales[-days:]):
            writer.writerow([day["date"], day["weekday"], units, prices.get(day["wm_yr_wk"], "")])
    return out


def load_series(path=SERIES_FILE):
    """(units sold per day, weekday index per day with Monday = 0, average shelf price)."""
    rows = list(csv.DictReader(open(path)))
    units = np.array([int(r["units_sold"]) for r in rows])
    weekdays = np.array([WEEKDAYS.index(r["weekday"]) for r in rows])
    price = float(np.mean([float(r["sell_price"]) for r in rows if r["sell_price"]]))
    return units, weekdays, price


# ----------------------------------------------------------------------------- problem

class RetailInventory:
    """Single-product inventory MDP with empirical, weekday-dependent demand (costs, gamma 0.9)."""

    action_names = None  # actions are order quantities 0..capacity

    def __init__(self, units, weekdays, price, capacity=None, unit_cost_fraction=0.7, holding=0.05,
                 fixed_order=1.0, gamma=0.9, weekly=True, start_stock=0, start_weekday=0):
        units, weekdays = np.asarray(units, dtype=int), np.asarray(weekdays, dtype=int)
        self.capacity = int(np.percentile(units, 99)) if capacity is None else int(capacity)
        self.price, self.unit_cost = float(price), unit_cost_fraction * float(price)
        self.holding, self.fixed_order, self.gamma = float(holding), float(fixed_order), float(gamma)
        self.weekly = weekly
        self.n_days = 7 if weekly else 1
        self.n_stock = self.capacity + 1
        self.n_states, self.n_actions = self.n_stock * self.n_days, self.capacity + 1
        # Demand distribution for each weekday (or one pooled distribution), over 0..max sold.
        groups = weekdays if weekly else np.zeros_like(weekdays)
        self.demand_pmf = np.zeros((self.n_days, units.max() + 1))
        for day in range(self.n_days):
            self.demand_pmf[day] = np.bincount(units[groups == day], minlength=units.max() + 1)
            self.demand_pmf[day] /= self.demand_pmf[day].sum()
        self._demand_cdf = np.cumsum(self.demand_pmf, axis=1)
        self._keep_reachable(start_weekday * self.n_stock + start_stock if weekly else start_stock)
        self.start = int(self.encode(start_stock, start_weekday if weekly else 0))

    def _keep_reachable(self, start):
        """Number only the (stock, day) cells reachable from the start cell (cell = day * n_stock + stock)."""
        reached, frontier = {start}, [start]
        while frontier:
            cell = frontier.pop()
            stock, day = cell % self.n_stock, cell // self.n_stock
            demands = np.flatnonzero(self.demand_pmf[day])
            for order in range(self.n_actions):
                _, left, next_day = self._outcome(stock, day, order, demands)
                for nxt in set((next_day * self.n_stock + left).tolist()) - reached:
                    reached.add(nxt)
                    frontier.append(nxt)
        self._cells = np.array(sorted(reached))  # state index -> cell
        self._state_of_cell = np.full(self.n_stock * self.n_days, -1)  # cell -> state index (-1: unreachable)
        self._state_of_cell[self._cells] = np.arange(len(self._cells))
        self.n_states = len(self._cells)

    @classmethod
    def from_m5(cls, path=SERIES_FILE, **settings):
        units, weekdays, price = load_series(path)
        return cls(units, weekdays, price, **settings)

    def encode(self, stock, day):
        """State index of (stock, day); -1 if that combination is unreachable."""
        return self._state_of_cell[np.asarray(day) * self.n_stock + np.asarray(stock)]

    def decode(self, states):
        cells = self._cells[np.asarray(states)]
        return cells % self.n_stock, cells // self.n_stock  # (stock, day)

    def _outcome(self, stock, day, order, demand):
        """Cost, units left and next day for given demands (works elementwise)."""
        on_shelf = np.minimum(stock + order, self.capacity)
        received = on_shelf - stock
        sold = np.minimum(on_shelf, demand)
        left = on_shelf - sold
        cost = (self.fixed_order * (received > 0) + self.unit_cost * received
                + self.holding * left - self.price * sold)
        return cost, left, (day + 1) % self.n_days

    def step(self, states, actions, rng):
        states, actions = np.broadcast_arrays(np.asarray(states, dtype=int), np.asarray(actions, dtype=int))
        if np.any((states < 0) | (states >= self.n_states)) or np.any((actions < 0) | (actions >= self.n_actions)):
            raise ValueError("state or action out of range")
        stock, day = self.decode(states)
        demand = (rng.random(states.shape)[..., None] > self._demand_cdf[day]).sum(axis=-1)
        cost, left, next_day = self._outcome(stock, day, actions, demand)
        return self.encode(left, next_day), cost, np.zeros(states.shape, dtype=bool)

    def step_one(self, state, action, rng):
        """One transition (the team's online estimator steps one sample at a time)."""
        stock, day = self.decode(state)
        demand = int((rng.random() > self._demand_cdf[day]).sum())
        cost, left, next_day = self._outcome(stock, day, action, demand)
        return int(self.encode(left, next_day)), float(cost), False

    def initial_states(self, count, rng):
        return np.full(count, self.start, dtype=int)

    def start_distribution(self):
        weights = np.zeros(self.n_states)
        weights[self.start] = 1.0
        return weights

    def exact_model(self):
        """Transition probabilities P (S, A, S) and expected one-step cost c (S, A)."""
        P, c = np.zeros((self.n_states, self.n_actions, self.n_states)), np.zeros((self.n_states, self.n_actions))
        demands = np.arange(self.demand_pmf.shape[1])
        for s in range(self.n_states):
            stock, day = self.decode(s)
            for a in range(self.n_actions):
                cost, left, next_day = self._outcome(stock, day, a, demands)
                c[s, a] = self.demand_pmf[day] @ cost
                np.add.at(P[s, a], self.encode(left, next_day), self.demand_pmf[day])
        return P, c


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extract", action="store_true", help=f"download M5 and rewrite {SERIES_FILE.name}")
    if parser.parse_args().extract:
        DATA_DIR.mkdir(exist_ok=True)
        print("wrote", extract_series(download_m5()))
    env = RetailInventory.from_m5()
    print(f"{ITEM} at {STORE}: capacity {env.capacity}, price ${env.price:.2f}, "
          f"{env.n_states} states x {env.n_actions} actions")


if __name__ == "__main__":
    main()
