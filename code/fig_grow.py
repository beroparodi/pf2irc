"""Figs. 3 and 6 of the paper: the recipe inverted (Fig. 3), and the clock (Fig. 6).

    python3 fig_grow.py     ->  fig18_grow.png (Fig. 3), fig_clock.png (Fig. 6), fig_grow_out.txt

Panels 1-2: DDO 154 and NGC 2915, observed curves, IRC today and at earlier epochs.
Panel 3   : their core radius against cosmic time; dash-dotted: growth ending earlier with feedback eps
            rising linearly in time, which gives the same core today (Sect. 5.4).
Fig. 6    : the clock, read as it is used: the end of growth t_acc (left axis, Gyr; right axis z_acc) against
            the v/r_c seen today (Eq. 24, inverted), for the nominal choices (median spin, eps = 0, n = 2), with
            the band of the 16 combinations of Table 1 (Lambda booked with k = 2/3 or 1, spin on top of E or
            inside it, eps = 0 or +0.009, n = 2 or 1.76; y_band.py), evaluated at every epoch: the choices do not
            depend on the epoch, Table 1 lists them at z = 2 and 0.5. The energy-removing layer, eps = -0.009, is
            left out: it has a P18 solution only for growth ending late (z <~ 0.1-0.8, depending on the other
            choices; Table 1, note a), so it cannot form a band. Ticks at the bottom: the observed v/r_c of the
            79 SPARC dwarfs (16-84 % range shaded, median dashed, its reading dotted). The two galaxies: their
            epoch is not measured but read off the clock; dot = reading at the median spin, horizontal bar =
            1-sigma error of v/r_c from the fit errors of v and r_c, vertical bar = epochs for spins between the
            16th and 84th percentile of the halo distribution (lambda = 0.035 exp(-+0.5)) together with that
            error. The choices of Table 1 are common to all galaxies and shift them together. Squares: the
            predictions for z = 2 and 3. Shaded at the top: growth ending after today. v/r_c is the same for all v.
Uses recipe.py (Sect. 5.3) and y_band.py (Table 1). Units: kpc, km/s; times in Gyr.
"""
import csv, io, contextlib
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import recipe as _g
with contextlib.redirect_stdout(io.StringIO()):          # y_band prints its table on import
    import y_band as yb

plt.rcParams.update({"font.size": 8.5, "savefig.bbox": "tight", "figure.dpi": 150})
C1, C2 = "#1f4e79", "#c0504d"
DPI, GYR, LAMSPIN = 300, 0.977792, 0.035
out = []
def report(s): print(s); out.append(s)
def t_of_z(z): return _g.t_of_z(z)*GYR                         # Gyr
def z_of_t(t): return brentq(lambda z: t_of_z(z)-t, 0, 1e4)
def _core(v, lam, tg): return _g.grow(v, lam, z_of_t(tg))[0]   # tg in Gyr
def _vc(v, rc, R): return _g.irc(np.maximum(R, 1e-9), v, rc)

_fits = {r["name"]: r for r in csv.DictReader(open("sparc_data/sparc79_fits.csv"))}
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.3), gridspec_kw={"width_ratios": [1.12, 1.12, 1]})
T0 = t_of_z(0); _tt = np.linspace(t_of_z(15), T0, 300)
zacc_gal = {}
for k, (gname, col, zalt) in enumerate([("DDO154", C1, 1.0), ("NGC2915", C2, 2.0)]):
    g = _fits[gname]; v_ = float(g["v"]); rc_ = float(g["rc"]); erc = float(g["erc"])
    zacc = brentq(lambda z: _g.grow(v_, LAMSPIN, z)[0]-rc_, 0, 10); tacc = t_of_z(zacc)
    zacc_gal[gname] = (zacc, v_/rc_, col)
    d = np.loadtxt(f"sparc_data/rotmod/{gname}_rotmod.dat")
    Rm = np.linspace(0, d[:, 0].max()*1.05, 300)
    for zz in [6, 2, 1]:
        if zz > zacc: ax[k].plot(Rm, _vc(v_, _core(v_, LAMSPIN, t_of_z(zz)), Rm), c="0.55", lw=0.8, ls="--")
    ax[k].plot(Rm, _vc(v_, rc_, Rm), c=col, lw=1.8)
    ax[k].errorbar(d[:, 0], d[:, 1], d[:, 2], fmt="o", ms=2.2, c="k", lw=0.7, elinewidth=0.7)
    _e = np.sqrt(d[:, 2]**2+(0.03*d[:, 1])**2+1.0)                 # errors as in the fits (Sect. 6.1, Table E.1)
    chi = np.sum(((d[:, 1]-_vc(v_, rc_, d[:, 0]))/_e)**2)/max(len(d)-2, 1)
    ax[k].set_title(f"{gname[:3]} {gname[3:]}, $z_{{\\rm acc}}={zacc:.2f}$", fontsize=8.5)
    ax[k].set_xlabel(r"$R$ [kpc]"); ax[k].set_ylim(0, None); ax[k].set_xlim(0, Rm[-1])
    ax[k].text(0.97, 0.05, "dashed: $z=" + ", ".join(str(z) for z in [6, 2, 1] if z > zacc) + "$",
               transform=ax[k].transAxes, ha="right", fontsize=7, color="0.4")
    tg = _tt[_tt <= tacc]; ax[2].plot(tg, [_core(v_, LAMSPIN, t) for t in tg], c=col, lw=1.8, label=f"{gname[:3]} {gname[3:]}")
    ax[2].plot([tacc, T0], [rc_, rc_], c=col, lw=1.2, ls=":")
    ax[2].axhspan(rc_-erc, rc_+erc, color=col, alpha=0.12, lw=0)
    tb = t_of_z(zalt); ef = brentq(lambda e: _g.grow(v_, LAMSPIN, zalt, e)[1]-rc_, 0, 0.2)
    _ta = np.linspace(tb, T0, 200)
    ax[2].plot(_ta, [_g.grow(v_, LAMSPIN, zalt, ef*(t-tb)/(T0-tb))[1] for t in _ta], c=col, lw=1.2, ls="-.")
    report(f"degeneracy: {gname} growth ending at z={zalt} (r_c={_g.grow(v_, LAMSPIN, zalt)[0]:.2f}) plus eps rising linearly to {ef:.4f} gives r_c={rc_:.2f} today")
    report(f"grown galaxies: {gname} v={v_:.1f} rc={rc_:.2f} v/rc={v_/rc_:.1f} -> z_acc={zacc:.3f} (t={tacc:.2f} Gyr), reduced chi2 of IRC={chi:.2f}")
ax[0].set_ylabel(r"$v_c$ [km s$^{-1}$]")
ax[2].set_xlabel(r"cosmic time $t$ [Gyr]"); ax[2].set_ylabel(r"$r_c$ [kpc]"); ax[2].set_xlim(0, T0); ax[2].set_ylim(0, None)
ax[2].legend(frameon=False, fontsize=7, loc="lower right", handlelength=1.4, handletextpad=0.5); ax[2].set_title(r"core growth, $\lambda=0.035$", fontsize=8.5)

# ---- panel 4: the clock, end of growth against the v/r_c seen today
fig2, a4 = plt.subplots(figsize=(3.46, 2.7))                # Fig. 6: single column
combos = [dict(k=k, s=s, eps=e, n=n) for k in (2/3, 1.0) for s in (0, 1) for e in (0.0, 0.009) for n in (2.0, 1.76)]
zz = np.concatenate([np.linspace(-0.45, 1, 146), np.linspace(1.01, 25, 400)])   # z < 0: growth ending after today
TT = np.array([t_of_z(z) for z in zz])
nomx = np.array([yb.vrc_model(z)[0] for z in zz])                                 # v/r_c rises with z
cx = [np.array([yb.vrc_model(z, **kw)[0] for z in zz]) for kw in combos]
xg = np.logspace(np.log10(5), np.log10(800), 300)
tq = lambda c: np.exp(np.interp(np.log(xg), np.log(c), np.log(TT), left=np.nan, right=np.nan))
tb = np.array([tq(c) for c in cx]); tlo, thi = np.min(tb, 0), np.max(tb, 0)
vr = np.array([float(r["v"])/float(r["rc"]) for r in _fits.values()])
p16, p50, p84 = np.percentile(vr, [16, 50, 84])
def zread(q, **kw):                                         # end of growth at which v/r_c = q (may be < 0)
    f = lambda z: yb.vrc_model(z, **kw)[0] - q
    return brentq(f, -0.45, 25)
a4.fill_between(xg, tlo, thi, color="0.85", lw=0, label="Table 1 choices")
a4.axvspan(p16, p84, color="#d08a1e", alpha=0.15, lw=0)
a4.axvline(p50, c="#d08a1e", lw=1.0, ls="--")
a4.plot(xg, tq(nomx), c="k", lw=1.6, label="nominal")
a4.plot(vr, np.full_like(vr, 0.84), "|", c="#d08a1e", ms=5, mew=0.7)
zmed = zread(p50); tmed = t_of_z(zmed)
a4.plot([p50, 800], [tmed, tmed], c="#d08a1e", lw=0.8, ls=":")
a4.text(780, tmed*0.95, f"$z={zmed:.2f}$", color="#a0650a", fontsize=7, ha="right", va="top")
a4.axhspan(T0, 30, color="0.55", alpha=0.3, lw=0); a4.axhline(T0, c="0.35", lw=0.8)
a4.text(780, T0*1.04, "after today", fontsize=6.5, color="0.25", ha="right", va="bottom")
lam16, lam84 = LAMSPIN*np.exp(-0.5), LAMSPIN*np.exp(0.5)
for gname, (za, vrg, col) in zacc_gal.items():
    g = _fits[gname]; sq = vrg*np.hypot(float(g["ev"])/float(g["v"]), float(g["erc"])/float(g["rc"]))
    z1, z2 = zread(vrg-sq, lam=lam16), zread(vrg+sq, lam=lam84); tn = t_of_z(za)
    a4.errorbar(vrg, tn, yerr=[[tn-t_of_z(z2)], [t_of_z(z1)-tn]], fmt="none", c=col, elinewidth=1.3, capsize=0, zorder=4)
    a4.errorbar(vrg, tn, xerr=sq, fmt="none", c=col, elinewidth=1.4, capsize=2.6, capthick=1.2, zorder=6)  # fit error, capped
    a4.plot(vrg, tn, "o", ms=3.0, c=col, mec="white", mew=0.5, zorder=7)
    zc = [zread(vrg, **kw) for kw in combos]
    report(f"clock reading {gname}: v/r_c = {vrg:.1f} +- {sq:.1f}; median spin z = {za:.2f} (measurement error alone "
           f"{zread(vrg-sq):.2f}-{zread(vrg+sq):.2f}); spins 16-84 % with the error {z1:.2f}-{z2:.2f}; choices of Table 1 "
           f"{min(zc):.2f}-{max(zc):.2f}")
_rat = [t_of_z(zread(zacc_gal["DDO154"][1], **kw))/t_of_z(zread(zacc_gal["NGC2915"][1], **kw)) for kw in combos]
report(f"clock reading: under each choice of Table 1, t_acc(DDO154)/t_acc(NGC2915) = {min(_rat):.2f}-{max(_rat):.2f}")
for zp in (2, 3):
    a4.plot(yb.vrc_model(zp)[0], t_of_z(zp), "s", ms=3.3, mfc="white", mec="k", mew=0.8, zorder=5)
a4.text(yb.vrc_model(2)[0]*1.2, t_of_z(2), "predicted", fontsize=7, ha="left", va="center")
a4.set_xscale("log"); a4.set_yscale("log"); a4.set_xlim(5, 800); a4.set_ylim(0.75, 19)
a4.set_xticks([10, 30, 100, 300]); a4.set_xticklabels(["10", "30", "100", "300"])
a4.minorticks_off(); a4.set_yticks([1, 2, 5, 10]); a4.set_yticklabels(["1", "2", "5", "10"])
a4.set_yticks([t for t in range(1, 14) if t not in (1, 2, 5, 10)], minor=True); a4.set_yticklabels([], minor=True)   # every Gyr up to today
a4.set_xlabel(r"$v/r_c$ today [km s$^{-1}$ kpc$^{-1}$]"); a4.set_ylabel(r"end of growth $t_{\rm acc}$ [Gyr]", labelpad=1)
a4r = a4.twinx(); a4r.set_yscale("log"); a4r.set_ylim(a4.get_ylim()); _zt = [0, 0.5, 1, 2, 3, 6]
a4r.set_yticks([t_of_z(z) for z in _zt]); a4r.set_yticklabels([str(z) for z in _zt]); a4r.minorticks_off()
a4r.set_ylabel(r"$z_{\rm acc}$", labelpad=1)
a4.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.89), handlelength=1.4, labelspacing=0.25, borderpad=0.1, borderaxespad=0.2)
_m = (zz > 0) & (zz < 4)
report(f"clock: nominal law t_acc ~ (v/r_c)^{np.polyfit(np.log(nomx[_m]), np.log(TT[_m]), 1)[0]:.2f} for 0 < z < 4; v/r_c for growth "
       f"ending today {yb.vrc_model(0)[0]:.1f} (choices of Table 1 {min(c[np.argmin(abs(zz))] for c in cx):.1f}-"
       f"{max(c[np.argmin(abs(zz))] for c in cx):.1f}); dwarfs below the nominal value: {(vr < yb.vrc_model(0)[0]).sum()}")
def _parts(z):                                              # t, y, lag factor and v/r_c at the end of growth z
    t = yb.t_of_z(z); return t, yb.vrc_model(z)[1], 1+yb.CL*yb.L3*t*t/yb.K**2, yb.vrc_model(z)[0]
_sl = []
for _z in (6, 3, 2, 1, 0.64, 0.3, 0.0):                      # local logarithmic slopes, by finite differences
    _a, _b = _parts(_z*1.001+0.001), _parts(_z*0.999-0.001); _dl = np.log(_a[0]/_b[0])
    _sv = np.log(_a[3]/_b[3])/_dl
    _sl.append(f"z={_z}: dln y/dln t {np.log(_a[1]/_b[1])/_dl:+.2f}, lag {np.log(_a[2]/_b[2])/_dl:+.2f}, dln t/dln(v/r_c) {1/_sv:+.2f}")
report("clock: local slopes (Sect. 6.4): " + "; ".join(_sl))
_gx = []
for _z in (2, 1, 0.5):                                       # growth exponents of r_c and rho(0), and of the cosmic mean (Sect. 5.1)
    _a, _b = _parts(_z*1.001+0.001), _parts(_z*0.999-0.001); _k = -np.log(_a[3]/_b[3])/np.log(_a[0]/_b[0])
    _H = 0.07*np.sqrt(0.3*(1+_z)**3+0.7)                     # H in km/s/kpc; t in kpc/(km/s)
    _gx.append(f"z={_z}: r_c ~ t^{_k:.2f}, rho(0) ~ t^{-2*_k:.2f}, cosmic mean ~ t^{-3*yb.t_of_z(_z)*_H:.2f}")
report("growth with Lambda (Sect. 5.1): " + "; ".join(_gx))
_l84 = LAMSPIN*np.exp(0.5)
report(f"clock floor at z = 2 (Sect. 9): nominal {yb.vrc_model(2)[0]:.0f}; spin at the 84th percentile {yb.vrc_model(2, lam=_l84)[0]:.0f}; "
       f"lowest choice of Table 1 {min(yb.vrc_model(2, **kw)[0] for kw in combos):.0f}; both {min(yb.vrc_model(2, lam=_l84, **kw)[0] for kw in combos):.0f} km/s/kpc")
_r = zacc_gal["NGC2915"][1]/zacc_gal["DDO154"][1]
report(f"clock: 2^-0.8 = {2**-0.8:.2f}; NGC 2915 has {_r:.2f} times the v/r_c of DDO 154 and stopped growing at "
       f"{t_of_z(zacc_gal['NGC2915'][0])/t_of_z(zacc_gal['DDO154'][0]):.2f} of its time ({_r:.2f}^-0.82 = {_r**-0.82:.2f})")
for _n in ("NGC1705", "NGC4214"):
    if _n in _fits:
        _q = float(_fits[_n]["v"])/float(_fits[_n]["rc"]); report(f"clock: {_n} v/r_c = {_q:.0f} reads z = {zread(_q):.1f}")
fig.tight_layout(w_pad=0.6)
_d = 0.03                                  # panel 2 has no axis label: move it left and widen panel 3 by the same amount
_p1, _p2 = ax[1].get_position(), ax[2].get_position()
ax[1].set_position([_p1.x0-_d, _p1.y0, _p1.width, _p1.height])
ax[2].set_position([_p2.x0-_d, _p2.y0, _p2.width+_d, _p2.height])
fig.savefig("fig18_grow.png", dpi=DPI); plt.close(fig)
fig2.tight_layout(); fig2.savefig("fig_clock.png", dpi=DPI); plt.close(fig2)

report(f"clock: observed v/r_c of the 79 dwarfs 16/50/84 % = {p16:.1f}/{p50:.1f}/{p84:.1f} km/s/kpc; "
       f"nominal reading of the median: z_acc = {zmed:.2f} (t = {t_of_z(zmed):.2f} Gyr)")
report(f"clock: nominal v/r_c at z = 2 and 3: {yb.vrc_model(2)[0]:.0f} and {yb.vrc_model(3)[0]:.0f}; band of Table 1 "
       f"at z = 2: {np.nanmin([yb.vrc_model(2, **kw)[0] for kw in combos]):.0f}-{np.nanmax([yb.vrc_model(2, **kw)[0] for kw in combos]):.0f}, "
       f"at z = 3: {np.nanmin([yb.vrc_model(3, **kw)[0] for kw in combos]):.0f}-{np.nanmax([yb.vrc_model(3, **kw)[0] for kw in combos]):.0f}")
open("fig_grow_out.txt", "w").write("\n".join(out) + "\n")
