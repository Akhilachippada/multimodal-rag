"""Generate demo data: sample_data/sales.csv, a chart image and (if PyMuPDF is installed) a small report PDF."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

out = Path(__file__).resolve().parent.parent / "sample_data"
out.mkdir(exist_ok=True)
rng = np.random.default_rng(7)

# ---- dataset
dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
n = 900
df = pd.DataFrame({
    "date": rng.choice(dates, n),
    "region": rng.choice(["North", "South", "East", "West"], n, p=[.3, .2, .3, .2]),
    "product": rng.choice(["Laptop", "Phone", "Tablet", "Watch"], n),
    "units": rng.integers(1, 30, n),
})
price = {"Laptop": 900, "Phone": 600, "Tablet": 400, "Watch": 250}
df["revenue"] = df["units"] * df["product"].map(price) * rng.uniform(.9, 1.1, n)
df["revenue"] = df["revenue"].round(2)
df.sort_values("date").to_csv(out / "sales.csv", index=False)

# ---- chart image
m = df.assign(month=pd.to_datetime(df["date"]).dt.to_period("Q").astype(str)).groupby("month")["revenue"].sum()
fig, ax = plt.subplots(figsize=(6, 3.6), dpi=130)
ax.bar(m.index, m.values, color="#3b6fb6")
ax.set_title("Quarterly revenue"); ax.set_ylabel("Revenue"); plt.xticks(rotation=40)
fig.tight_layout(); fig.savefig(out / "quarterly_revenue.png"); plt.close(fig)

# ---- report PDF
try:
    import fitz
except ImportError:
    print("PyMuPDF not installed - skipped PDF")
else:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 545, 250),
        "Annual Sales Report 2025\n\nRevenue grew steadily through 2025, driven by laptop and phone sales in the North "
        "and East regions. Figure 1 shows quarterly revenue. Table 1 lists the top products by revenue. "
        "The West region remained the smallest market, and management plans a marketing push there next year.",
        fontsize=11)
    page.insert_image(fitz.Rect(50, 260, 400, 470), filename=str(out / "quarterly_revenue.png"))
    page.insert_text((50, 485), "Figure 1: Quarterly revenue chart (bar graph).", fontsize=9)
    top = df.groupby("product")["revenue"].sum().sort_values(ascending=False)
    x0, y0, w, h = 50, 520, 150, 22
    rows = [("Product", "Revenue")] + [(k, f"{v:,.0f}") for k, v in top.items()]
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            rect = fitz.Rect(x0 + c * w, y0 + r * h, x0 + (c + 1) * w, y0 + (r + 1) * h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            page.insert_text((rect.x0 + 5, rect.y1 - 7), val, fontsize=10)
    page.insert_text((50, y0 + len(rows) * h + 16), "Table 1: Revenue by product.", fontsize=9)
    doc.save(out / "sales_report.pdf")
    print("wrote", out / "sales_report.pdf")
print("wrote sample data in", out)
