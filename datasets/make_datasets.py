"""
Generate every dataset used in the roadmap. Fully offline and reproducible (fixed seeds).

    python datasets/make_datasets.py

Outputs (all inside datasets/):
    customers.csv, products.csv, orders.csv, order_items.csv   e-commerce star schema
    shop.db                                                     same tables in SQLite (used by the SQL module)
    churn.csv            binary classification (telecom-style churn) with missing values + dirty categories
    housing.csv          regression (house prices) with outliers, missing values, non-linear effects
    reviews.csv          short product reviews with sentiment label (NLP)
    sensors.csv          hourly IoT time series with trend, seasonality, anomalies
    ab_test.csv          A/B test on conversion + revenue (statistics module)
    messy.csv            deliberately dirty table for data-cleaning practice
    knowledge_base/*.md  small company knowledge base (RAG / agents / MCP modules)

The data are SYNTHETIC. The relationships inside them are real (planted on purpose) so models and
statistical tests have something genuine to find, but they are not real customers or real houses.
Every module also points at real public datasets you can move on to.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 42


# --------------------------------------------------------------------------------------
# E-commerce star schema
# --------------------------------------------------------------------------------------
def make_shop(rng: np.random.Generator):
    n_cust = 2_000
    countries = ["US", "UK", "DE", "FR", "IN", "BD", "CA", "AU"]
    country_p = [0.30, 0.12, 0.10, 0.08, 0.15, 0.08, 0.10, 0.07]
    signup = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n_cust), unit="D")
    customers = pd.DataFrame(
        {
            "customer_id": np.arange(1, n_cust + 1),
            "first_name": rng.choice(
                ["Ava", "Liam", "Noah", "Mia", "Zoe", "Omar", "Priya", "Chen", "Lena", "Sara", "Ravi", "Tom"], n_cust
            ),
            "country": rng.choice(countries, n_cust, p=country_p),
            "signup_date": signup.date,
            "age": np.clip(rng.normal(36, 11, n_cust).round(), 18, 80).astype(int),
            "is_premium": rng.random(n_cust) < 0.18,
        }
    )

    cats = {
        "Electronics": (["Headphones", "Keyboard", "Webcam", "Speaker", "Monitor", "Mouse"], 80, 600),
        "Home": (["Lamp", "Blender", "Cushion", "Kettle", "Rug", "Clock"], 15, 200),
        "Fashion": (["T-Shirt", "Jacket", "Sneakers", "Backpack", "Watch", "Scarf"], 10, 180),
        "Books": (["Novel", "Cookbook", "Biography", "Textbook", "Comic", "Atlas"], 8, 60),
        "Sports": (["Yoga Mat", "Dumbbell", "Bottle", "Tent", "Racket", "Helmet"], 12, 250),
    }
    rows = []
    pid = 1
    for cat, (names, lo, hi) in cats.items():
        for nm in names:
            for variant in ("Basic", "Pro"):
                price = round(float(rng.uniform(lo, hi)) * (1.6 if variant == "Pro" else 1.0), 2)
                rows.append((pid, f"{nm} {variant}", cat, price, round(price * rng.uniform(0.45, 0.75), 2)))
                pid += 1
    products = pd.DataFrame(rows, columns=["product_id", "product_name", "category", "unit_price", "unit_cost"])

    # orders: seasonality (Nov/Dec peak), premium customers order more
    n_orders = 12_000
    weights = customers["is_premium"].map({True: 3.0, False: 1.0}).to_numpy()
    cust_ids = rng.choice(customers["customer_id"], n_orders, p=weights / weights.sum())
    day = rng.integers(0, 730, n_orders)
    dates = pd.to_datetime("2023-06-01") + pd.to_timedelta(day, unit="D")
    month_boost = dates.month.map({11: 2.0, 12: 2.4, 1: 0.8, 2: 0.8}).fillna(1.0).to_numpy()
    keep = rng.random(n_orders) < (month_boost / month_boost.max())
    dates, cust_ids = dates[keep], cust_ids[keep]
    # an order cannot precede the customer's signup
    signup_map = pd.to_datetime(customers.set_index("customer_id")["signup_date"])
    dates = pd.DatetimeIndex(np.maximum(dates.to_numpy(), signup_map.loc[cust_ids].to_numpy()))
    orders = pd.DataFrame(
        {
            "order_id": np.arange(1, len(dates) + 1),
            "customer_id": cust_ids,
            "order_date": dates.date,
            "status": rng.choice(["delivered", "shipped", "cancelled", "returned"], len(dates), p=[0.82, 0.08, 0.06, 0.04]),
            "channel": rng.choice(["web", "mobile", "store"], len(dates), p=[0.5, 0.38, 0.12]),
        }
    ).sort_values("order_date", ignore_index=True)
    orders["order_id"] = np.arange(1, len(orders) + 1)

    items = []
    for oid in orders["order_id"]:
        k = int(rng.choice([1, 2, 3, 4], p=[0.55, 0.28, 0.12, 0.05]))
        for p in rng.choice(products["product_id"], k, replace=False):
            items.append((oid, int(p), int(rng.choice([1, 2, 3], p=[0.8, 0.15, 0.05]))))
    order_items = pd.DataFrame(items, columns=["order_id", "product_id", "quantity"])
    order_items.insert(0, "item_id", np.arange(1, len(order_items) + 1))
    return customers, products, orders, order_items


# --------------------------------------------------------------------------------------
# Churn (classification)
# --------------------------------------------------------------------------------------
def make_churn(rng: np.random.Generator) -> pd.DataFrame:
    n = 5_000
    tenure = np.clip(rng.exponential(28, n).round(), 0, 72).astype(int)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["Fiber", "DSL", "None"], n, p=[0.44, 0.34, 0.22])
    monthly = (
        20
        + np.where(internet == "Fiber", 45, np.where(internet == "DSL", 25, 0))
        + rng.normal(0, 6, n)
    ).clip(18, 130).round(2)
    support_calls = rng.poisson(np.where(internet == "Fiber", 2.2, 1.2))
    paperless = rng.random(n) < 0.6
    autopay = rng.random(n) < 0.45
    senior = rng.random(n) < 0.16
    total = (monthly * np.maximum(tenure, 1) * rng.uniform(0.95, 1.05, n)).round(2)

    logit = (
        -1.0
        + 1.4 * (contract == "Month-to-month")
        - 1.2 * (contract == "Two year")
        - 0.035 * tenure
        + 0.018 * (monthly - 60)
        + 0.30 * support_calls
        + 0.35 * paperless
        - 0.5 * autopay
        + 0.25 * senior
        + 0.5 * (internet == "Fiber")
        + rng.normal(0, 0.6, n)
    )
    p = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < p).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{100000 + i}" for i in range(n)],
            "tenure_months": tenure,
            "contract": contract,
            "internet_service": internet,
            "monthly_charges": monthly,
            "total_charges": total,
            "support_calls": support_calls,
            "paperless_billing": np.where(paperless, "Yes", "No"),
            "autopay": np.where(autopay, "Yes", "No"),
            "senior_citizen": senior.astype(int),
            "churned": churn,
        }
    )
    # realistic mess: missing values + inconsistent casing
    df.loc[rng.random(n) < 0.03, "total_charges"] = np.nan
    df.loc[rng.random(n) < 0.02, "internet_service"] = np.nan
    flip = rng.random(n) < 0.04
    df.loc[flip, "contract"] = df.loc[flip, "contract"].str.lower()
    return df


# --------------------------------------------------------------------------------------
# Housing (regression)
# --------------------------------------------------------------------------------------
def make_housing(rng: np.random.Generator) -> pd.DataFrame:
    n = 3_000
    hoods = {"Riverside": 1.25, "Downtown": 1.45, "Hillcrest": 1.10, "Oakwood": 0.95, "Eastgate": 0.80, "Lakeview": 1.30}
    neighborhood = rng.choice(list(hoods), n)
    area = np.clip(rng.lognormal(7.3, 0.35, n), 450, 6_000).round()
    bedrooms = np.clip((area / 650 + rng.normal(0, 0.8, n)).round(), 1, 7).astype(int)
    bathrooms = np.clip((bedrooms * 0.6 + rng.normal(0, 0.5, n)).round(1), 1, 6)
    age = rng.integers(0, 90, n)
    garage = rng.choice([0, 1, 2, 3], n, p=[0.12, 0.35, 0.43, 0.10])
    dist = np.clip(rng.gamma(2.5, 3.0, n), 0.3, 40).round(1)
    quality = rng.integers(1, 11, n)
    mult = np.array([hoods[h] for h in neighborhood])
    price = (
        45_000
        + area * 95 * mult
        + bedrooms * 4_000
        + bathrooms * 9_000
        + garage * 11_000
        + (quality - 5) * 14_000
        - age * 850
        - dist * 2_200
        + (area / 1000) ** 2 * 18_000
    )
    price = price * rng.lognormal(0, 0.07, n)
    df = pd.DataFrame(
        {
            "neighborhood": neighborhood,
            "area_sqft": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age,
            "garage_spaces": garage,
            "distance_to_center_km": dist,
            "quality_score": quality,
            "price": price.round(-2),
        }
    )
    df.loc[rng.random(n) < 0.04, "age_years"] = np.nan
    df.loc[rng.random(n) < 0.03, "quality_score"] = np.nan
    out = rng.choice(n, 15, replace=False)  # a few absurd listings (data-entry errors)
    df.loc[out, "price"] = df.loc[out, "price"] * rng.choice([0.05, 8], 15)
    return df


# --------------------------------------------------------------------------------------
# Reviews (NLP)
# --------------------------------------------------------------------------------------
def make_reviews(rng: np.random.Generator) -> pd.DataFrame:
    pos_open = ["Absolutely love this", "Great quality", "Works perfectly", "Really happy with this", "Excellent value", "Fantastic purchase"]
    neg_open = ["Very disappointed", "Terrible quality", "Stopped working", "Not worth the money", "Awful experience", "Would not recommend"]
    neu_open = ["It is okay", "Average product", "Does the job", "Decent but nothing special", "Mixed feelings"]
    pos_mid = ["and it arrived early", "and the build feels solid", "and setup took two minutes", "and it looks even better in person", "and customer support was helpful"]
    neg_mid = ["and it broke after a week", "and the packaging was damaged", "and support never replied", "and it does not match the description", "and the battery dies quickly"]
    neu_mid = ["although shipping was slow", "but the price is fair", "and it is fine for the price", "but I expected a bit more", "and nothing stands out"]
    cats = ["Electronics", "Home", "Fashion", "Books", "Sports"]
    rows = []
    for _ in range(4_000):
        s = rng.choice(["positive", "negative", "neutral"], p=[0.45, 0.30, 0.25])
        o, m = {
            "positive": (pos_open, pos_mid),
            "negative": (neg_open, neg_mid),
            "neutral": (neu_open, neu_mid),
        }[s]
        text = f"{rng.choice(o)} {rng.choice(m)}."
        if rng.random() < 0.3:  # sprinkle in cross-sentiment noise so the task is not trivial
            other = rng.choice(pos_mid + neg_mid + neu_mid)
            text += f" Also, {other}."
        stars = {"positive": rng.choice([4, 5], p=[0.35, 0.65]), "negative": rng.choice([1, 2], p=[0.6, 0.4]), "neutral": 3}[s]
        rows.append((text, s, int(stars), rng.choice(cats)))
    return pd.DataFrame(rows, columns=["review_text", "sentiment", "stars", "category"])


# --------------------------------------------------------------------------------------
# Sensors (time series)
# --------------------------------------------------------------------------------------
def make_sensors(rng: np.random.Generator) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=24 * 180, freq="h")
    t = np.arange(len(idx))
    temp = 15 + 0.004 * t + 6 * np.sin(2 * np.pi * t / 24 - 2) + 8 * np.sin(2 * np.pi * t / (24 * 365) * 1.0) + rng.normal(0, 0.8, len(t))
    power = 200 + 30 * np.sin(2 * np.pi * t / 24 - 1) + 15 * (idx.dayofweek < 5) + rng.normal(0, 5, len(t))
    anomalies = rng.choice(len(t), 25, replace=False)
    power[anomalies] += rng.choice([-1, 1], 25) * rng.uniform(60, 120, 25)
    df = pd.DataFrame({"timestamp": idx, "temperature_c": temp.round(2), "power_kw": power.round(2), "is_anomaly": 0})
    df.loc[anomalies, "is_anomaly"] = 1
    df.loc[rng.choice(len(t), 60, replace=False), "temperature_c"] = np.nan  # sensor dropouts
    return df


# --------------------------------------------------------------------------------------
# A/B test
# --------------------------------------------------------------------------------------
def make_ab(rng: np.random.Generator) -> pd.DataFrame:
    n = 8_000
    group = rng.choice(["control", "treatment"], n)
    device = rng.choice(["mobile", "desktop"], n, p=[0.6, 0.4])
    base = np.where(device == "mobile", 0.085, 0.12)
    lift = np.where(group == "treatment", 1.12, 1.0)  # true relative lift 12%
    converted = rng.random(n) < base * lift
    revenue = np.where(converted, rng.lognormal(3.6, 0.5, n) * np.where(group == "treatment", 1.03, 1.0), 0.0)
    return pd.DataFrame(
        {"user_id": np.arange(1, n + 1), "group": group, "device": device, "converted": converted.astype(int), "revenue": revenue.round(2)}
    )


# --------------------------------------------------------------------------------------
# Deliberately messy data
# --------------------------------------------------------------------------------------
def make_messy(rng: np.random.Generator) -> pd.DataFrame:
    n = 600
    names = rng.choice(["  alice smith", "BOB JONES", "Carla Diaz ", "dave  lee", "Eve Adams", "frank o'neil"], n)
    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "name": names,
            "signup": rng.choice(["2024-01-15", "15/01/2024", "Jan 15, 2024", "2024.02.03", "not available", ""], n),
            "age": rng.choice(["25", "31", "47", "N/A", "-1", "999", "38", "52", ""], n),
            "salary": rng.choice(["$52,000", "61000", "48,500", "$75k", "", "unknown", "90000"], n),
            "country": rng.choice(["US", "usa", "U.S.", "United States", "UK", "United Kingdom", "india", "IN", "Bangladesh"], n),
        }
    )
    dup = df.sample(40, random_state=1)
    return pd.concat([df, dup], ignore_index=True).sample(frac=1, random_state=2).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Knowledge base for RAG / agents / MCP
# --------------------------------------------------------------------------------------
KB = {
    "returns_policy.md": """# Returns Policy
Customers may return most items within 30 days of delivery for a full refund.
Electronics have a 15 day return window and must include original packaging.
Books, opened software and personalised items cannot be returned.
Refunds are issued to the original payment method within 5 to 7 business days after the return is inspected.
Premium members get free return shipping; other customers pay a flat 6 dollar label fee.""",
    "shipping.md": """# Shipping
Standard shipping takes 3 to 5 business days and costs 4.99 dollars. Orders above 50 dollars ship free.
Express shipping takes 1 to 2 business days and costs 12.99 dollars.
We ship to the US, UK, Germany, France, India, Bangladesh, Canada and Australia.
International orders may incur customs fees which are paid by the customer.
Tracking numbers are emailed once the parcel leaves the warehouse.""",
    "premium_membership.md": """# Premium Membership
Premium costs 79 dollars per year. Benefits: free express shipping, free returns, early access to sales and a dedicated support line.
Membership renews automatically each year; cancel any time from the account page and keep benefits until the term ends.
Premium members earn double loyalty points on every purchase.""",
    "warranty.md": """# Warranty
All electronics carry a 12 month limited warranty covering manufacturing defects.
Damage from drops, water or unauthorised repairs is not covered.
To claim, contact support with your order number and a short description or photo of the fault.
Approved claims are repaired or replaced within 10 business days.""",
    "payment.md": """# Payment Methods
We accept Visa, Mastercard, American Express, PayPal and bank transfer.
Buy-now-pay-later is available for orders above 100 dollars in the US and UK only.
Payment is charged when the order ships, not when it is placed.
If a payment fails we retry once after 24 hours and then cancel the order.""",
    "privacy.md": """# Privacy
We store only the data needed to fulfil orders: name, address, email and order history.
We never sell personal data. Customers can request export or deletion of their data by emailing privacy support.
Deletion requests are completed within 30 days. Payment card numbers are never stored on our servers.""",
    "support_hours.md": """# Support Hours
Chat support is available 24/7. Phone support runs Monday to Friday, 9am to 6pm Eastern Time.
Premium members have a dedicated phone line with a target wait time under 2 minutes.
Average first response time on email is 4 hours. Complaints are escalated to a team lead within 24 hours.""",
    "loyalty_points.md": """# Loyalty Points
Every dollar spent earns 1 point; premium members earn 2 points per dollar.
100 points can be redeemed for a 5 dollar discount. Points expire 12 months after they are earned.
Points are not earned on shipping fees, taxes or gift cards.""",
    "cancellations.md": """# Order Cancellations
Orders can be cancelled free of charge until they ship. After shipping, please use the returns process instead.
To cancel, open Orders in your account and choose Cancel. Refunds for cancelled orders appear within 3 business days.""",
    "sustainability.md": """# Sustainability
Packaging is 90 percent recycled cardboard and we ship carbon neutral within the US and UK.
The refurbished programme resells returned electronics after full testing with a 6 month warranty.""",
}


def write_kb():
    kb = HERE / "knowledge_base"
    kb.mkdir(exist_ok=True)
    for name, text in KB.items():
        (kb / name).write_text(text + "\n", encoding="utf-8")


def main():
    rng = np.random.default_rng(SEED)
    customers, products, orders, items = make_shop(rng)
    customers.to_csv(HERE / "customers.csv", index=False)
    products.to_csv(HERE / "products.csv", index=False)
    orders.to_csv(HERE / "orders.csv", index=False)
    items.to_csv(HERE / "order_items.csv", index=False)

    db = HERE / "shop.db"
    if db.exists():
        db.unlink()
    with sqlite3.connect(db) as con:
        customers.to_sql("customers", con, index=False)
        products.to_sql("products", con, index=False)
        orders.to_sql("orders", con, index=False)
        items.to_sql("order_items", con, index=False)

    make_churn(rng).to_csv(HERE / "churn.csv", index=False)
    make_housing(rng).to_csv(HERE / "housing.csv", index=False)
    make_reviews(rng).to_csv(HERE / "reviews.csv", index=False)
    make_sensors(rng).to_csv(HERE / "sensors.csv", index=False)
    make_ab(rng).to_csv(HERE / "ab_test.csv", index=False)
    make_messy(rng).to_csv(HERE / "messy.csv", index=False)
    write_kb()
    for f in sorted(HERE.glob("*.csv")) + [db]:
        print(f"{f.name:18s} {f.stat().st_size / 1024:8.1f} KB")
    print(f"knowledge_base/    {len(KB)} docs")


if __name__ == "__main__":
    main()
