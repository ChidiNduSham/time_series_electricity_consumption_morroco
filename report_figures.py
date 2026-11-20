"""Generates the report charts and a results.json file for the Week 5 Data Science report.
Uses the same aggregation and statistical choices as Week5_Data_Science_Analysis.ipynb."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from itertools import combinations
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.multitest import multipletests

for f in ["Poppins-Regular","Poppins-Medium","Poppins-Bold","Poppins-Light","Poppins-Italic"]:
    fm.fontManager.addfont(f"/usr/share/fonts/truetype/google-fonts/{f}.ttf")
NAVY,BRIGHT,TEAL,AQUA,LAQ,PANEL,INK,GRID="#20578B","#42C2FF","#2FC9BB","#B8FFF9","#EFFFFD","#F7F7F7","#2E2E2E","#DCE5EA"
plt.rcParams.update({"font.family":"Poppins","font.size":8,"text.color":INK,"axes.labelcolor":INK,
  "xtick.color":INK,"ytick.color":INK,"axes.edgecolor":GRID,"axes.facecolor":PANEL,"figure.facecolor":PANEL,
  "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":GRID,"grid.linewidth":.7,
  "axes.axisbelow":True,"legend.frameon":False,"savefig.facecolor":PANEL,"axes.titleweight":"medium","axes.titlesize":9})
Z=["PowerConsumption_Zone1","PowerConsumption_Zone2","PowerConsumption_Zone3"]; ZN=["Zone 1","Zone 2","Zone 3"]
ZC=dict(zip(Z,[NAVY,BRIGHT,TEAL])); ENV=["Temperature","Humidity","WindSpeed"]
ENVL={"Temperature":"Temperature","Humidity":"Humidity","WindSpeed":"Wind speed"}
df=pd.read_csv("data/powerconsumption.csv"); df["Datetime"]=pd.to_datetime(df["Datetime"],format="%m/%d/%Y %H:%M")
df=df.sort_values("Datetime").copy(); df["Hour"]=df.Datetime.dt.hour; df["Month"]=df.Datetime.dt.month
df["DayOfWeek"]=df.Datetime.dt.day_name(); df["Date"]=df.Datetime.dt.date
ti=df.set_index("Datetime"); daily=ti[Z+ENV].resample("D").mean()
order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
MN=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
def save(fig,name): fig.savefig(f"figs/{name}.png",dpi=220,bbox_inches="tight",pad_inches=.08); plt.close(fig)
def leg(ax,**k): ax.legend(handles=[plt.Line2D([],[],color=ZC[z],lw=2,label=n) for z,n in zip(Z,ZN)],ncol=3,loc="upper left",**k)

# F1 distribution
fig,ax=plt.subplots(1,2,figsize=(6.2,2.7),gridspec_kw={"width_ratios":[1.25,1]})
for z,n in zip(Z,ZN):
    ax[0].hist(df[z],bins=60,density=True,histtype="stepfilled",alpha=.38,color=ZC[z]); ax[0].hist(df[z],bins=60,density=True,histtype="step",color=ZC[z],lw=1.3,label=n)
ax[0].set(title="Distribution of 10-minute readings",xlabel="Recorded value",yticks=[]); ax[0].legend(loc="upper right")
bp=ax[1].boxplot([df[z] for z in Z],orientation="horizontal",tick_labels=ZN,patch_artist=True,showmeans=True,widths=.55,flierprops=dict(marker=".",markersize=1.5,alpha=.5,markeredgecolor="#8FA3B3"),
   meanprops=dict(marker="D",markerfacecolor="white",markeredgecolor=INK,markersize=4),medianprops=dict(color=INK,lw=1.4),whiskerprops=dict(color=INK,lw=.8),capprops=dict(color=INK,lw=.8))
for p,z in zip(bp["boxes"],Z): p.set(facecolor=ZC[z],alpha=.8,edgecolor=INK,lw=.8)
ax[1].set(title="Spread: median (line), mean (diamond)",xlabel="Recorded value"); ax[1].invert_yaxis(); ax[1].grid(axis="y",visible=False)
for a in ax: a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v/1000:.0f}k"))
fig.tight_layout(); save(fig,"f1_distribution")

# F2 daily trend
fig,ax=plt.subplots(figsize=(6.2,2.9))
for z,n in zip(Z,ZN): ax.plot(daily.index,daily[z],color=ZC[z],lw=1.1,label=n)
ax.axvspan(pd.Timestamp("2017-08-28"),pd.Timestamp("2017-09-02"),color=AQUA,alpha=.9,lw=0)
ax.annotate("Zone 3 step-down\n(28 Aug – 1 Sep)",xy=(pd.Timestamp("2017-08-31"),17000),xytext=(pd.Timestamp("2017-09-12"),4600),fontsize=7,arrowprops=dict(arrowstyle="-",color=INK,lw=.7),ha="left")
ax.set(title="Daily mean recorded consumption, 1 Jan – 30 Dec 2017",ylabel="Daily mean recorded value",ylim=(3000,42000)); leg(ax)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v/1000:.0f}k")); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b")); ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
fig.tight_layout(); save(fig,"f2_daily")

# F3 hourly
hp=df.groupby("Hour")[Z].mean()
fig,ax=plt.subplots(figsize=(6.2,2.9)); ax.axvspan(17.5,22.5,color=AQUA,alpha=.9,lw=0); ax.text(20,9300,"Evening peak\n18:00 – 22:00",ha="center",fontsize=7,bbox=dict(fc=AQUA,ec="none",pad=1.5))
for z,n in zip(Z,ZN):
    ax.plot(hp.index,hp[z],color=ZC[z],lw=1.8,marker="o",ms=3.2,label=n); h=hp[z].idxmax(); ax.plot(h,hp[z].max(),"o",ms=7,mfc="none",mec=ZC[z],mew=1.4)
ax.set(title="Mean recorded consumption by hour of day",xlabel="Hour of day",ylabel="Mean recorded value",xticks=range(0,24,2),ylim=(8000,47000)); leg(ax)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v/1000:.0f}k")); fig.tight_layout(); save(fig,"f3_hourly")

# F4 weekday
wp=df.groupby("DayOfWeek")[Z].mean().reindex(order); dev=(wp/wp.mean()-1)*100
fig,ax=plt.subplots(1,2,figsize=(6.2,2.7)); x=np.arange(7); w=.27
for i,z in enumerate(Z): ax[0].bar(x+(i-1)*w,wp[z],w,color=ZC[z],label=ZN[i])
ax[0].set(title="Mean recorded value by day",xticks=x,xticklabels=[d[:3] for d in order]); ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v/1000:.0f}k")); ax[0].legend(ncol=3,loc="upper center",fontsize=7,bbox_to_anchor=(.5,1.0),columnspacing=1)
ax[0].set_ylim(0,40000)
for z,n in zip(Z,ZN): ax[1].plot(x,dev[z],color=ZC[z],lw=1.8,marker="o",ms=3.5,label=n)
ax[1].axhline(0,color=INK,lw=.8); ax[1].set(title="Deviation from each zone's weekly mean (%)",xticks=x,xticklabels=[d[:3] for d in order])
fig.tight_layout(); save(fig,"f4_weekday")

# F5 monthly
mp=df.groupby("Month")[Z].mean()
fig,ax=plt.subplots(figsize=(6.2,2.9)); ax.axvspan(6.5,8.5,color=AQUA,alpha=.9,lw=0)
for z,n in zip(Z,ZN): ax.plot(mp.index,mp[z],color=ZC[z],lw=1.8,marker="o",ms=3.5,label=n)
ax.set(title="Monthly mean recorded consumption",xticks=range(1,13),xticklabels=MN,ylabel="Mean recorded value",ylim=(9000,40000)); leg(ax)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v/1000:.0f}k")); fig.tight_layout(); save(fig,"f5_monthly")

# F6 STL
stl={z:STL(daily[z].dropna(),period=7,robust=True).fit() for z in Z}
fig,axs=plt.subplots(4,3,figsize=(6.2,5.6),sharex="col")
for j,z in enumerate(Z):
    r=stl[z]
    for i,(s,t) in enumerate([(r.observed,"Observed"),(r.trend,"Trend"),(r.seasonal,"Seasonal (7-day)"),(r.resid,"Residual")]):
        a=axs[i,j]; a.plot(s.index,s.values,color=ZC[z],lw=1 if i!=3 else .7)
        if i==3: a.axhline(0,color=INK,lw=.6)
        if j==0: a.set_ylabel(t,fontsize=7)
        if i==0: a.set_title(ZN[j])
        a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v/1000:.0f}k" if abs(v)>=1000 else f"{v:.0f}")); a.tick_params(labelsize=6.5)
        a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b")); a.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=3))
fig.tight_layout(); save(fig,"f6_stl")

# correlations (daily, Spearman) – identical to notebook
rows=[]
for z in Z:
    for v in ENV:
        rho,p=stats.spearmanr(daily[v],daily[z],nan_policy="omit"); rows.append(dict(zone=z,var=v,rho=rho,p=p))
cor=pd.DataFrame(rows); cor["holm"]=multipletests(cor.p,method="holm")[1]
# F7 scatter
fig,ax=plt.subplots(1,3,figsize=(6.2,2.6),sharey=True)
for a,v in zip(ax,ENV):
    for z in Z: a.scatter(daily[v],daily[z],s=7,alpha=.5,color=ZC[z],lw=0)
    a.set(xlabel={"Temperature":"Temperature (daily mean)","Humidity":"Humidity (daily mean)","WindSpeed":"Wind speed (daily mean)"}[v]); a.set_title(ENVL[v])
ax[0].set_ylabel("Daily mean recorded value"); ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v/1000:.0f}k")); leg(ax[0],fontsize=6.5)
ax[0].legend(handles=[plt.Line2D([],[],marker="o",ls="",color=ZC[z],label=n,ms=4) for z,n in zip(Z,ZN)],loc="lower right",fontsize=6.5)
fig.tight_layout(); save(fig,"f7_scatter")

# comparative
fr=stats.friedmanchisquare(*(daily[z].dropna() for z in Z)); n=daily[Z].dropna().shape[0]; W=fr.statistic/(n*2)
pw=[]
for a,b in combinations(Z,2):
    t=stats.wilcoxon(daily[a],daily[b]); pw.append(dict(a=a,b=b,stat=t.statistic,p=t.pvalue,gt=int((daily[a]>daily[b]).sum())))
pwd=pd.DataFrame(pw); pwd["holm"]=multipletests(pwd.p,method="holm")[1]
cv=df[Z].std()/df[Z].mean()
fig,ax=plt.subplots(1,2,figsize=(6.2,2.6))
bp=ax[0].boxplot([daily[z] for z in Z],tick_labels=ZN,patch_artist=True,widths=.55,medianprops=dict(color=INK,lw=1.4),whiskerprops=dict(color=INK,lw=.8),capprops=dict(color=INK,lw=.8),flierprops=dict(marker=".",markersize=3,markeredgecolor=GRID))
for p,z in zip(bp["boxes"],Z): p.set(facecolor=ZC[z],alpha=.8,edgecolor=INK,lw=.8)
ax[0].set(title="Daily mean consumption (n = 364 days)"); ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v/1000:.0f}k")); ax[0].grid(axis="x",visible=False)
b=ax[1].bar(ZN,cv.values,color=[ZC[z] for z in Z],width=.55); ax[1].set(title="Coefficient of variation (SD ÷ mean)",ylim=(0,.46))
for r,v,s in zip(b,cv.values,df[Z].std().values): ax[1].text(r.get_x()+r.get_width()/2,v+.01,f"{v:.2f}\nSD {s:,.0f}",ha="center",fontsize=7)
ax[1].grid(axis="x",visible=False); fig.tight_layout(); save(fig,"f8_zones")

# F9 heatmaps hour x month
cmap=LinearSegmentedColormap.from_list("t",[ "#F7F7F7",AQUA,BRIGHT,NAVY])
fig,axs=plt.subplots(1,3,figsize=(6.2,2.8),sharey=True)
vmin,vmax=df[Z].mean().min()*.3,df[Z].max().max()*.82
for a,z,nm in zip(axs,Z,ZN):
    g=df.pivot_table(index="Hour",columns="Month",values=z,aggfunc="mean"); im=a.imshow(g.values,aspect="auto",cmap=cmap,vmin=10000,vmax=40000,origin="lower")
    a.set_title(nm); a.set_xticks([0,3,6,9]); a.set_xticklabels(["Jan","Apr","Jul","Oct"]); a.grid(False)
    for s in a.spines.values(): s.set_visible(False)
axs[0].set_ylabel("Hour of day"); axs[0].set_yticks(range(0,24,3))
cb=fig.colorbar(im,ax=axs,shrink=.8,pad=.02,aspect=22); cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v/1000:.0f}k")); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
save(fig,"f9_heatmap")

# rings
def ring(pct,name,col):
    fig,ax=plt.subplots(figsize=(1.3,1.3)); fig.patch.set_alpha(0); ax.set_facecolor("none")
    ax.pie([pct,100-pct] if pct<100 else [100],colors=[col,"#E8E8E8"] if pct<100 else [col],startangle=90,counterclock=False,wedgeprops=dict(width=.2,edgecolor="none"))
    ax.text(0,0,f"{pct:g}%",ha="center",va="center",fontsize=13,fontweight="medium"); ax.set_aspect("equal")
    fig.savefig(f"assets/{name}.png",dpi=250,transparent=True,bbox_inches="tight",pad_inches=.02); plt.close(fig)
ring(100,"ring_complete",BRIGHT); ring(100,"ring_interval",NAVY); ring(round(364/365*100,1),"ring_coverage",BRIGHT)

# results.json
sm=df[Z].agg(["count","mean","median","min","max","std","var"]).T; sm["Q1"]=df[Z].quantile(.25); sm["Q3"]=df[Z].quantile(.75); sm["IQR"]=sm.Q3-sm.Q1; sm["CV"]=sm["std"]/sm["mean"]; sm["Skew"]=df[Z].skew()
res=dict(summary=sm.round(4).to_dict(orient="index"),hourly=hp.round(2).to_dict(orient="index"),weekday=wp.round(2).to_dict(orient="index"),
 monthly=mp.round(2).to_dict(orient="index"),corr=cor.round(6).to_dict(orient="records"),friedman=dict(stat=fr.statistic,p=fr.pvalue,W=W,n=n),pairwise=pwd.to_dict(orient="records"),
 stl={z:dict(Fs=float(max(0,1-stl[z].resid.var()/(stl[z].resid+stl[z].seasonal).var())),Ft=float(max(0,1-stl[z].resid.var()/(stl[z].resid+stl[z].trend).var()))) for z in Z},
 acf1={z:float(daily[z].autocorr(1)) for z in Z})
json.dump(res,open("results.json","w"),indent=1,default=float); print("done"); print(pwd); print(cor.round(4))
