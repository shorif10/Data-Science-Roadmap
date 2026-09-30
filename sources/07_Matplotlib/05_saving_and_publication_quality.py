# %% [markdown]
# # Saving Figures and Publication Quality
#
# **Why this matters.** A chart eventually leaves your notebook: it goes into a slide, a PDF report, a wiki page, a paper. A blurry PNG, a cut-off label, or a red-green palette that 1 in 12 men
# cannot read turns good analysis into a bad impression. Knowing how to export crisply, choose accessible colours and describe a chart in words is a small skill with a large payoff.
#
# **You will learn**
# - save with `fig.savefig`, and relate **dpi**, figure size and pixel size
# - choose **raster** (PNG) vs **vector** (SVG/PDF) and verify the difference
# - use `bbox_inches="tight"`, transparent backgrounds and sensible fonts
# - pick a **colour-blind-safe palette**, check contrast, and add redundant encodings (line styles, markers, direct labels)
# - write alt text and save metadata
#
# **Prerequisites:** [Subplots, layout, annotations and styling](03_subplots_layout_annotations_styling.ipynb).
#
# > Data used: `datasets/orders.csv` (synthetic). Files are written only to a temporary directory that is deleted at the end.

# %%
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.max_open_warning"] = 0
DATA = Path("../datasets")
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
by_channel = orders.groupby([orders.order_date.dt.to_period("M").dt.to_timestamp(), "channel"]).size().unstack("channel")

out = tempfile.TemporaryDirectory()
OUT = Path(out.name)


def channel_chart(colors=None, styles=None, figsize=(7, 3.6)):
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    for i, ch in enumerate(by_channel.columns):
        kw = {}
        if colors:
            kw["color"] = colors[i]
        if styles:
            kw["linestyle"], kw["marker"] = styles[i]
        ax.plot(by_channel.index, by_channel[ch], label=ch, **kw)
    ax.set(title="Orders per month by channel", xlabel="Month", ylabel="Orders (count)")
    ax.legend(title="Channel")
    return fig, ax


fig, ax = channel_chart()
plt.show()

# %% [markdown]
# ## 1. `savefig`, dpi and pixels
# `fig.savefig(path)` writes the figure; the format comes from the extension. For raster formats (PNG, JPG), **pixels = inches x dpi**. A 7 x 3.6 inch figure at 100 dpi is 700 x 360 pixels (fine on a screen, soft in print); at 300 dpi it is 2100 x 1080.

# %%
for dpi in (72, 100, 300):
    fig.savefig(OUT / f"chart_{dpi}.png", dpi=dpi)
    img = plt.imread(OUT / f"chart_{dpi}.png")
    size_kb = (OUT / f"chart_{dpi}.png").stat().st_size / 1024
    print(f"dpi {dpi:3d}: image array {img.shape[1]} x {img.shape[0]} px, file {size_kb:5.1f} KB   (expected {int(7 * dpi)} x {int(3.6 * dpi)})")

# %% [markdown]
# *What just happened?* We exported the same figure three times and read the PNG back to confirm its real pixel size: it matches inches x dpi. Higher dpi means more pixels and a larger file. Rules of thumb: **screen/slides 100-150 dpi, print 300 dpi**.
# Set the physical size to match where it will be used (a column of a paper is about 3.3 inches wide), then let dpi handle sharpness; scaling a small figure up later blurs it.

# %% [markdown]
# ## 2. Raster vs vector
# **Raster** (PNG, JPG) stores a grid of pixels: sharp only at the resolution it was made. **Vector** (SVG, PDF) stores *drawing instructions* (lines, text) that stay sharp at any zoom and usually stay editable.
# Use vector for line charts, bar charts and anything with text that will be resized or printed; use PNG for huge scatter plots (a vector file with a million points is enormous) and where the target does not support vector.
# JPG is lossy and smears thin lines and text: avoid it for charts.

# %%
for ext in ("png", "svg", "pdf"):
    fig.savefig(OUT / f"chart.{ext}", dpi=150)
    print(f"{ext}: {(OUT / f'chart.{ext}').stat().st_size / 1024:6.1f} KB")

svg = (OUT / "chart.svg").read_text(encoding="utf-8")
pdf_head = (OUT / "chart.pdf").read_bytes()[:5]
print("SVG starts with:", svg[:5], "| PDF starts with:", pdf_head)
print("number of <text> elements in the default SVG:", svg.count("<text"))

# %% [markdown]
# By default Matplotlib converts text in SVG to *paths* (shapes) so fonts always render, which is why the count of `<text>` elements above is what it is. Set `plt.rcParams["svg.fonttype"] = "none"` to keep real text
# (then the viewer needs the font), and `plt.rcParams["pdf.fonttype"] = 42` to embed editable TrueType text in PDFs (journals often require it).

# %%
plt.rcParams["svg.fonttype"] = "none"
fig.savefig(OUT / "chart_text.svg")
print("number of <text> elements with svg.fonttype='none':", (OUT / "chart_text.svg").read_text(encoding="utf-8").count("<text"))
plt.rcParams["svg.fonttype"] = "path"

# %% [markdown]
# ## 3. Cut-off labels, transparency, metadata
# If labels are clipped in the saved file (they look fine in the notebook because Jupyter crops for display), use `bbox_inches="tight"`: it crops to the content. With `layout="constrained"` this is often unnecessary,
# but it is the standard safety net. `transparent=True` removes the white background (handy for dark slides, but be careful: dark text on a dark slide vanishes). `metadata` embeds information in the file.

# %%
fig.savefig(OUT / "tight.png", dpi=150, bbox_inches="tight", metadata={"Title": "Orders per month by channel", "Description": "Line chart: monthly orders for web, mobile and store; all peak in Nov-Dec."})
from matplotlib.image import imread  # noqa: E402

print("tight png shape:", imread(OUT / "tight.png").shape)
try:
    from PIL import Image  # PIL ships with matplotlib's dependencies
    print("embedded PNG text chunks:", Image.open(OUT / "tight.png").text)
except Exception as e:  # noqa: BLE001
    print("could not read metadata:", e)

# %% [markdown]
# ## 4. Colour that everyone can read
# About 1 in 12 men and 1 in 200 women have some form of colour-vision deficiency, most often red-green. The default Matplotlib cycle is reasonable, but a red/green pair, or a rainbow colormap, is a classic trap.
# The **Okabe-Ito palette** was designed to stay distinguishable for the common types; its eight hex codes are widely published.

# %%
OKABE_ITO = {"orange": "#E69F00", "sky_blue": "#56B4E9", "bluish_green": "#009E73", "yellow": "#F0E442",
             "blue": "#0072B2", "vermillion": "#D55E00", "reddish_purple": "#CC79A7", "black": "#000000"}

fig_p, ax_p = plt.subplots(figsize=(8, 1.2), layout="constrained")
ax_p.bar(range(8), [1] * 8, color=list(OKABE_ITO.values()))
ax_p.set_xticks(range(8), list(OKABE_ITO), rotation=30, ha="right")
ax_p.set(title="Okabe-Ito colour-blind-safe palette", xlabel="Colour name", ylabel="Swatch")
ax_p.set_yticks([])
plt.show()

# %% [markdown]
# **Do not rely on colour alone.** Add a second cue: line style, marker shape, direct labels. A quick test is to convert to **grayscale**: if two lines become the same grey, colour is doing all the work. Compute perceived brightness (luminance) to check:

# %%
def luminance(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast_vs_white(hex_color):
    return (1.0 + 0.05) / (luminance(hex_color) + 0.05)


table = pd.DataFrame({"hex": OKABE_ITO.values(), "luminance": [luminance(c) for c in OKABE_ITO.values()],
                      "contrast_vs_white": [contrast_vs_white(c) for c in OKABE_ITO.values()]}, index=OKABE_ITO.keys()).round(2)
table

# %% [markdown]
# The contrast ratio follows the WCAG formula (a ratio of 3:1 or more is the usual minimum for graphics against their background, 4.5:1 for normal text). Yellow (`#F0E442`) is nearly invisible on a white background: use it only on dark backgrounds or for fills, never for thin lines.
# Pick lines from the higher-contrast entries and add redundant encoding:

# %%
cols = [OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["bluish_green"]]
styles = [("-", "o"), ("--", "s"), (":", "^")]
fig_a, ax_a = channel_chart(colors=cols, styles=styles)
for ln in ax_a.lines:
    ln.set_markevery(3)
plt.show()
print("distinct (colour, linestyle) pairs:", len({(ln.get_color(), ln.get_linestyle()) for ln in ax_a.lines}))

# %% [markdown]
# Now each series differs by colour **and** dash pattern **and** marker: readable in greyscale print and for colour-blind readers. Even better is to label lines directly at their end and drop the legend (a technique used in the Seaborn storytelling notebook).
# For continuous data use a perceptually uniform colormap such as `viridis` (the default), not `jet`/rainbow, whose bright bands create fake "edges".

# %% [markdown]
# ## 5. Publication checklist
# - Size the figure to its final physical size; fonts then come out at the size you chose (8-10 pt in print).
# - Export vector (PDF/SVG) for line art, PNG at 300 dpi for dense plots; keep the script that makes the figure.
# - Title, axis labels with units, legend or direct labels, a source note for external data.
# - Accessible colours plus a second cue; check contrast.
# - **Alt text**: one or two sentences saying what the chart shows *and the takeaway*. A screen-reader user gets this instead of the picture. Example for our chart: "Line chart of monthly orders for web, mobile and store channels from June 2023 to May 2025. All three channels peak in December 2024; web has the most orders overall."
# Verify the alt-text claim rather than assuming it:

# %%
peak_by_channel = by_channel.idxmax()
highest_overall = by_channel.sum().idxmax()
print("month of peak per channel:", {k: f"{v:%b %Y}" for k, v in peak_by_channel.items()})
print("channel with most orders overall:", highest_overall)

# %% [markdown]
# Finally, a small helper that enforces the checklist (formats, dpi, tight bbox) so every export is consistent:

# %%
def save_figure(fig, name, outdir, formats=("png", "svg"), dpi=300):
    """Save `fig` as outdir/name.<fmt> for each format; return the list of written Paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = outdir / f"{name}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    return paths


print([p.name for p in save_figure(fig_a, "orders_by_channel", OUT / "exports")])

# %% [markdown]
# ## Common mistakes
#
# 1. **Saving a small figure at low dpi, then enlarging it** in a slide: blurry. Size first, dpi second, or use vector.
# 2. **JPG for charts**: lossy, smeared lines.
# 3. **Clipped labels** in saved files: `bbox_inches="tight"` or constrained layout.
# 4. **Red/green contrast** as the only difference, or a rainbow colormap for continuous data.
# 5. **Yellow lines on white** (contrast ratio near 1).
# 6. **Relying on the legend colour alone** with six similar colours: use direct labels and a second encoding.
# 7. **No alt text**, or alt text that only repeats the title instead of the takeaway.
# 8. **Saving with `plt.savefig` after `plt.show()`** in scripts: `show` may close the figure, leaving an empty save. Save first, or save from the `fig` object.

# %%
# Mistake 8: savefig after a figure is closed (what show() does in some backends) produces an empty canvas, not an error
f, a = plt.subplots(figsize=(3, 2))
a.plot([1, 2, 3])
f.savefig(OUT / "before_close.png")
plt.close("all")
g, b = plt.subplots(figsize=(3, 2))                      # a NEW empty figure is now "current"
plt.savefig(OUT / "wrong_figure.png")                  # pyplot saves the *current* figure: the empty one
same = plt.imread(OUT / "before_close.png").shape == plt.imread(OUT / "wrong_figure.png").shape
diff = float(np.abs(plt.imread(OUT / "before_close.png") - plt.imread(OUT / "wrong_figure.png")).sum())
print("same size:", same, "| total pixel difference:", round(diff, 1), "(non-zero: they are different pictures)")

# %% [markdown]
# ## Interview corner
#
# **1. PNG or SVG for a chart in a report?**
# <details><summary>Answer</summary>
# SVG/PDF (vector) for line art and text-heavy charts: they stay sharp at any size and file sizes are small. PNG at 300 dpi for very dense plots (many thousands of points) or targets that do not accept vector.
# </details>
#
# **2. How do you check your palette works for colour-blind readers?**
# <details><summary>Answer</summary>
# Use a designed safe palette (Okabe-Ito, viridis for continuous), add a second encoding (line style, marker, direct label), check luminance/contrast, and preview with a colour-vision simulator or greyscale.
# </details>
#
# **3. A saved figure has cut-off axis labels. Why, and what do you do?**
# <details><summary>Answer</summary>
# The notebook display crops automatically but the saved canvas is exactly the figure size. Use <code>bbox_inches="tight"</code>, or <code>layout="constrained"</code>, or adjust the figure margins.
# </details>
#
# **4. What goes in alt text for a chart?**
# <details><summary>Answer</summary>
# Chart type, what is plotted (variables, period, units) and the main takeaway, in one or two sentences. Decorative detail is not needed; the conclusion is.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Sharp at print size
# Create a figure `fig1` of **6 x 3 inches** showing the monthly total orders (all channels) with title and labels, and save it to `OUT / "ex1.png"` at **dpi=200**. The check reads the file and verifies its pixel size.

# %% tags=["starter"]
total = by_channel.sum(axis=1)
fig1, ax1 = plt.subplots()
# draw, then save to OUT / "ex1.png" with dpi=200

# %%
img = plt.imread(OUT / "ex1.png")
assert img.shape[:2] == (600, 1200), f"6x3 inches at 200 dpi should be 1200 x 600 px (width x height), got {img.shape[1]} x {img.shape[0]}; set figsize=(6, 3) and dpi=200 (do not use bbox_inches='tight', it changes the size)"
assert np.array_equal(ax1.lines[0].get_ydata(), total.values), "plot the total monthly orders"
assert ax1.get_title() and ax1.get_xlabel() and "order" in ax1.get_ylabel().lower()
print("Exercise 1 passed")

# %% tags=["solution"]
total = by_channel.sum(axis=1)
fig1, ax1 = plt.subplots(figsize=(6, 3), layout="constrained")
ax1.plot(total.index, total.values)
ax1.set(title="Total orders per month", xlabel="Month", ylabel="Orders (count)")
fig1.savefig(OUT / "ex1.png", dpi=200)

# %% [markdown]
# ### Exercise 2: Vector exports
# Save `fig1` as **SVG and PDF** into `OUT` (`ex2.svg`, `ex2.pdf`). Then set `smaller` to the name of the format (`"png"`, `"svg"` or `"pdf"`) with the **smallest file** among `ex1.png` (from exercise 1), `ex2.svg`, `ex2.pdf`, measured from the files.

# %% tags=["starter"]
smaller = ...

# %%
assert (OUT / "ex2.svg").read_text(encoding="utf-8").lstrip().startswith("<?xml"), "ex2.svg should be a real SVG file"
assert (OUT / "ex2.pdf").read_bytes().startswith(b"%PDF"), "ex2.pdf should be a real PDF file"
sizes = {"png": (OUT / "ex1.png").stat().st_size, "svg": (OUT / "ex2.svg").stat().st_size, "pdf": (OUT / "ex2.pdf").stat().st_size}
assert smaller == min(sizes, key=sizes.get), f"measured sizes (bytes): {sizes}"
print("Exercise 2 passed:", sizes)

# %% tags=["solution"]
fig1.savefig(OUT / "ex2.svg")
fig1.savefig(OUT / "ex2.pdf")
sizes = {"png": (OUT / "ex1.png").stat().st_size, "svg": (OUT / "ex2.svg").stat().st_size, "pdf": (OUT / "ex2.pdf").stat().st_size}
smaller = min(sizes, key=sizes.get)

# %% [markdown]
# ### Exercise 3: Accessible lines
# Build `fig3, ax3` with the three channel lines using the Okabe-Ito colours `blue`, `vermillion`, `bluish_green` **and** three different line styles, so the lines differ in colour *and* dash pattern. Include legend, title, axis labels.
# *Hint: reuse `channel_chart(colors=..., styles=...)`, or write the loop yourself.*

# %% tags=["starter"]
fig3, ax3 = plt.subplots()
# your code here

# %%
assert len(ax3.lines) == 3
cols = [ln.get_color().lower() for ln in ax3.lines]
allowed = {OKABE_ITO["blue"].lower(), OKABE_ITO["vermillion"].lower(), OKABE_ITO["bluish_green"].lower()}
assert set(cols) == allowed, f"use exactly these three Okabe-Ito colours: {sorted(allowed)}; got {cols}"
assert len({ln.get_linestyle() for ln in ax3.lines}) == 3, "each line needs a different linestyle"
assert ax3.get_legend() is not None and ax3.get_title() and ax3.get_xlabel() and "order" in ax3.get_ylabel().lower()
print("Exercise 3 passed")

# %% tags=["solution"]
fig3, ax3 = channel_chart(colors=[OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["bluish_green"]],
                          styles=[("-", "o"), ("--", "s"), (":", "^")])

# %% [markdown]
# ### Exercise 4: A reliable export helper
# Write `export_all(fig, stem, outdir)` that saves `fig` as `<stem>.png` (dpi 300), `<stem>.svg` and `<stem>.pdf` inside `outdir` (create it if needed) with a tight bounding box, and returns a dict `{format: Path}`.
# The check exports `fig3` into a new folder and verifies the files.

# %% tags=["starter"]
def export_all(fig, stem, outdir):
    ...

# %%
res = export_all(fig3, "accessible_channels", OUT / "ex4_exports")
assert set(res) == {"png", "svg", "pdf"}, "return a dict with keys png, svg, pdf"
assert all(Path(p).exists() and Path(p).parent == OUT / "ex4_exports" for p in res.values()), "files must exist inside outdir"
h, w = plt.imread(res["png"]).shape[:2]
wi, hi = fig3.get_size_inches()
assert w >= int(wi * 300 * 0.85) and h >= int(hi * 300 * 0.85), f"PNG should be about figsize x 300 dpi ({int(wi*300)} x {int(hi*300)}), got {w} x {h}"
assert Path(res["pdf"]).read_bytes().startswith(b"%PDF") and Path(res["svg"]).read_text(encoding="utf-8").lstrip().startswith("<?xml")
print("Exercise 4 passed")

# %% tags=["solution"]
def export_all(fig, stem, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for fmt in ("png", "svg", "pdf"):
        p = outdir / f"{stem}.{fmt}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths[fmt] = p
    return paths

# %% [markdown]
# ### Exercise 5 (open-ended): Accessibility review
# Take any chart from this module and audit it: (1) is colour the only cue? (2) are all contrast ratios at least 3:1 against white (use `contrast_vs_white`)? (3) write alt text with the takeaway and **verify each claim in it with a line of code**.
# Redesign one thing that failed the audit.

# %% [markdown]
# ## Clean up

# %%
plt.close("all")
out.cleanup()
print("temporary export folder removed:", not OUT.exists())

# %% [markdown]
# ## Recap
# - `fig.savefig(path, dpi=...)`: pixels = inches x dpi; size for the destination, 300 dpi for print; verify by reading the file back.
# - Vector (SVG/PDF) for line art and text; PNG for dense plots; avoid JPG. `bbox_inches="tight"` prevents clipped labels; fonttype settings control editable text.
# - Use designed palettes (Okabe-Ito, viridis), check contrast, and add a second cue (linestyle, marker, direct label).
# - Write alt text that states the takeaway, and verify its claims from the data.
# - Wrap exports in a helper so every figure is saved the same way.
#
# **Go deeper:** Matplotlib docs "Customizing Matplotlib with style sheets and rcParams", the `savefig` API page, and the Okabe-Ito palette description on the Color Universal Design site by Okabe and Ito.
#
# **Next module:** [Seaborn: distributions](../08_Seaborn/01_distributions.ipynb)
