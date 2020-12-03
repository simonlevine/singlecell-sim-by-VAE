import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.style.use("ggplot")

ann = sc.read_h5ad(fp)
d = ann.obs.groupby(["Ventilated", "cell_type_coarse"]).apply(lambda _df: len(_df)).to_frame().reset_index().rename(columns={0: "count"})
for c in ["Ventilated", "cell_type_coarse"]:
    d[c] = d[c].astype(str)

g = sns.catplot(
    data=d,
    x="cell_type_coarse",
    y="count",
    hue="Ventilated",
    kind="bar",
)
g.despine(left=True)
g.set_axis_labels("Cell Type", "Cell Count")
g.legend.set_title("")
plt.xticks(rotation=90)