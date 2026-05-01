#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ratios = [0.01, 0.25, 0.50, 0.75, 0.99]

em_bad    = [55/1440, 192/1440, 315/1440, 470/1440, 608/1440]
em_terr   = [11/1440,  89/1440, 153/1440, 244/1440, 255/1440]
gp_bad    = [38/1440,  24/1440,  40/1440,  54/1440, 177/1440]
gp_terr   = [13/1440,   1/1440,   6/1440,   9/1440,  40/1440]

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(ratios, em_bad,  "o-",  color="#e05c2e", lw=2, ms=7, label="EM — bad+terrible")
ax.plot(ratios, em_terr, "o--", color="#e05c2e", lw=1.5, ms=5, alpha=0.7, label="EM — terrible only")
ax.plot(ratios, gp_bad,  "s-",  color="#2e7de0", lw=2, ms=7, label="GP — bad+terrible")
ax.plot(ratios, gp_terr, "s--", color="#2e7de0", lw=1.5, ms=5, alpha=0.7, label="GP — terrible only")

ax.set_xlabel("Incorrect-data ratio (r)", fontsize=12)
ax.set_ylabel("Misalignment rate", fontsize=12)
ax.set_title("Misalignment vs training ratio\nEM sweep vs GP (terrible-99) sweep", fontsize=13)
ax.set_xticks(ratios)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.set_ylim(bottom=0)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()

out = "plots/sweep_misalignment.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")
