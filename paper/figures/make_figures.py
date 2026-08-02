import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})
ROME_C, VERS_C, GRID = "#d1495b", "#2e6f95", "#8d99ae"
FAM = {"v2":"#8d99ae", "v_n":"#2e6f95", "rot":"#2a9d8f", "full":"#d1495b"}

# ================= FIG 1 · E7 sequential editing (flagship) =================
k = np.array([10,20,40,60])
rome = {"eff":[1.00,0.95,0.88,0.82], "par":[0.50,0.40,0.35,0.43], "loc":[1.00,1.00,0.98,0.92]}
vers = {"eff":[0.90,0.80,0.85,0.88], "par":[0.80,0.60,0.58,0.62], "loc":[1.00,1.00,0.98,0.98]}
load2 = {"eff":1.00,"par":0.70,"loc":0.98}
titles = [("eff","Efficacy"),("par","Paraphrase"),("loc","Locality")]
fig, ax = plt.subplots(1,3, figsize=(7.0,2.35), constrained_layout=True)
for a,(key,ttl) in zip(ax, titles):
    a.plot(k, rome[key], "o-", color=ROME_C, lw=1.6, ms=4.5, label="ROME (rank-one)")
    a.plot(k, vers[key], "s-", color=VERS_C, lw=1.6, ms=4.5, label="Versor (load 3)")
    a.plot([60],[load2[key]], "*", color=VERS_C, ms=12, mec="k", mew=.4, label="Versor (load 2)")
    a.set_title(ttl); a.set_xlabel("accumulated edits $k$"); a.set_ylim(0.0,1.06)
    a.set_xticks(k)
ax[0].set_ylabel("score")
ax[0].legend(loc="lower left", frameon=False, fontsize=7)
fig.savefig("/home/claude/fig_editing.pdf", bbox_inches="tight")
fig.savefig("/home/claude/fig_editing.png", bbox_inches="tight")

# ================= FIG 2 · E2 sparse capacity (linear scaling) =================
M = np.array([1,2,4,8,16,32,64])
hashc  = np.array([4,8,16,32,48,128,192])
anchor = np.array([6,8,16,24,48,64,128])
fig2, a = plt.subplots(figsize=(3.3,2.6), constrained_layout=True)
a.plot(M, hashc, "o-", color=VERS_C, lw=1.6, ms=4.5, label="hash addressing  $O(1)$")
a.plot(M, anchor, "s-", color="#2a9d8f", lw=1.6, ms=4.5, label="anchor addressing  $O(M)$")
a.plot(M, 3*M, "--", color=GRID, lw=1.2, label="linear guide  ($3M$)")
a.set_xlabel("number of slots $M$"); a.set_ylabel("total capacity (assoc.\\ @ $\\geq$90\\%)")
a.set_title("E2 · Sparse memory: linear scaling")
a.legend(loc="upper left", frameon=False)
fig2.savefig("/home/claude/fig_sparse.pdf", bbox_inches="tight")
fig2.savefig("/home/claude/fig_sparse.png", bbox_inches="tight")

# ================= FIG 3 · E1 capacity/dimension (sub-linear + encoding) =========
dims = np.array([16,32,64,128])
ratio = {"v2":[0.062,0.031,0.047,0.039], "v_n":[0.125,0.062,0.078,0.055],
         "rot":[0.188,0.094,0.078,0.039], "full":[0.125,0.094,0.016,0.016]}
names = {"v2":"grade-2 (v2)","v_n":"mixed-grade (v\\_n)","rot":"rotors","full":"full-spectrum"}
mk = {"v2":"o","v_n":"s","rot":"^","full":"D"}
fig3, a = plt.subplots(figsize=(3.3,2.6), constrained_layout=True)
for fam in ["v_n","rot","v2","full"]:
    a.plot(dims, ratio[fam], mk[fam]+"-", color=FAM[fam], lw=1.6, ms=4.5, label=names[fam])
a.set_xscale("log", base=2); a.set_xticks(dims); a.set_xticklabels(dims)
a.set_xlabel("algebra dimension $2^n$"); a.set_ylabel("capacity / dimension")
a.set_title("E1 · Single-field: sub-linear capacity")
a.legend(loc="upper right", frameon=False, ncol=1)
fig3.savefig("/home/claude/fig_capacity.pdf", bbox_inches="tight")
fig3.savefig("/home/claude/fig_capacity.png", bbox_inches="tight")
print("Figuras generadas: fig_editing, fig_sparse, fig_capacity (.pdf/.png)")
