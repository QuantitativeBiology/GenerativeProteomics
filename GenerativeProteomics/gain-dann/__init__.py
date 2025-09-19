import os
import sys
import matplotlib.pyplot as plt

# project_dir = os.getcwd()

# if project_dir not in sys.path:
#     sys.path.append(project_dir)

# plot_folder = f"{project_dir}/reports/"

# Matplotlib config
plt.rcParams["figure.figsize"] = [10, 8]

# Matplotlib set font to sans-times
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"

# Matplotlib set main axis font size
plt.rcParams["axes.titlesize"] = 7

# Grid
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "lightgray"
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["grid.linestyle"] = "-"

# Matplotlib omit top and right spines
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# Matplotlib set legend loc
plt.rcParams["legend.loc"] = "best"

# Matplotlib set axis below true
plt.rcParams["axes.axisbelow"] = True

PALETTE_COLORS = {
    "main": "#fc8b64",
    "secondary": "#909cc5",
}