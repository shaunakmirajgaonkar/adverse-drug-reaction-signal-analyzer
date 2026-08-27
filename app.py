
from pathlib import Path
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Adverse Drug Reaction Signal Analyzer",page_icon="💊",layout="wide")
ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"; ASSET=ROOT/"assets/adr_signal_analyzer_dashboard_visual.svg"

st.markdown("""<style>
.stApp{background:linear-gradient(180deg,#fbfdfc,#f2f8fb);color:#173248}.block-container{max-width:1540px;padding-top:1rem}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #dce7e5}[data-testid="stSidebar"] *{color:#173248!important}
.hero{background:linear-gradient(135deg,#effcf5,#eef7ff 55%,#fff4e7);border:1px solid #dce9e4;border-radius:26px;padding:28px 30px;margin-bottom:20px}
.eyebrow{font-size:.74rem;font-weight:850;letter-spacing:.14em;color:#267c55;text-transform:uppercase}
.hero h1{font-size:2.38rem;line-height:1.08;margin:.35rem 0 .5rem;color:#173248!important}.hero p{color:#5e7185;max-width:1000px;margin:0}
.pill{display:inline-block;background:#fff;border:1px solid #d9e6e1;border-radius:999px;padding:7px 12px;margin:11px 6px 0 0;font-size:.82rem;font-weight:750;color:#35566a}
.card{background:#fff;border:1px solid #dce7e5;border-radius:18px;padding:16px;box-shadow:0 8px 24px rgba(20,50,70,.04)}
.label{font-size:.74rem;text-transform:uppercase;letter-spacing:.06em;font-weight:800;color:#75869a}.value{font-size:1.9rem;font-weight:850;color:#173248;margin-top:4px}.sub{font-size:.78rem;color:#7b8b9d}
.section{font-size:1.18rem;font-weight:850;color:#18394e;margin:22px 0 10px}.warning{background:#fff7e9;border:1px solid #efddb9;border-radius:15px;padding:14px;color:#6b511f}
.footer{text-align:center;color:#7b8d9c;font-size:.76rem;margin-top:18px}
</style>""",unsafe_allow_html=True)

def normalize(df):
    x=df.copy(); aliases={"drug":"medicine_name","medicine":"medicine_name","reaction":"reaction_term","date":"report_date","completeness":"report_completeness_pct","onset":"onset_days"}
    cols=[]
    for c in x.columns:
        k=re.sub(r"[^a-z0-9]+","_",str(c).strip().lower()).strip("_"); cols.append(aliases.get(k,k))
    x.columns=cols; return x
def num(s,d=0): return pd.to_numeric(s,errors="coerce").fillna(d)
def severity_weight(s): return s.map({"Non-serious":20,"Serious":68,"Hospitalization":92}).fillna(20)

def score_signals(x):
    x=x.copy()
    x["onset_days"]=num(x["onset_days"]); x["report_completeness_pct"]=num(x["report_completeness_pct"]).clip(0,1)
    x["follow_up_events"]=num(x["follow_up_events"]).clip(0,5)
    x["seriousness_signal"]=severity_weight(x["seriousness"])
    x["timing_signal"]=(1-x["onset_days"].clip(0,30)/30)*100
    x["completeness_signal"]=100-x["report_completeness_pct"]*100
    x["followup_signal"]=x["follow_up_events"]/5*100
    x["signal_score"]=(.52*x.seriousness_signal+.18*x.timing_signal+.15*x.completeness_signal+.15*x.followup_signal).clip(0,100).round(1)
    x["signal_band"]=pd.cut(x.signal_score,[-.1,34.9,59.9,79.9,100.1],labels=["Low","Moderate","High","Critical"])
    return x

def aggregate(x):
    g=x.groupby(["medicine_id","medicine_name","drug_class"],as_index=False).agg(
        report_count=("report_id","count"),avg_signal=("signal_score","mean"),
        serious_reports=("seriousness",lambda s:int((s!="Non-serious").sum())),
        hospitalization_reports=("seriousness",lambda s:int((s=="Hospitalization").sum())),
        avg_completeness=("report_completeness_pct","mean"),median_onset_days=("onset_days","median"))
    g["recurrence_component"]=g.report_count.clip(0,100)
    g["seriousness_component"]=(g.serious_reports/g.report_count.replace(0,np.nan)*100).fillna(0)
    g["review_priority"]=(.43*g.avg_signal+.27*g.recurrence_component+.20*g.seriousness_component+.10*(100-g.avg_completeness*100)).clip(0,100).round(1)
    g["priority_band"]=pd.cut(g.review_priority,[-.1,34.9,59.9,79.9,100.1],labels=["Low","Moderate","High","Critical"])
    return g

st.sidebar.markdown("## 💊 SafeSignal Local")
st.sidebar.caption("Detect • Triage • Review")
page=st.sidebar.radio("Workspace",["Dashboard","Medicine Signals","Reaction Patterns","Seriousness Review","Trend Monitor","Review Queue","Scenario Lab","Reports & Export"],label_visibility="collapsed")
st.sidebar.divider()
u1=st.sidebar.file_uploader("Upload anonymized ADR reports CSV",type=["csv"])
u2=st.sidebar.file_uploader("Upload medicine-context CSV",type=["csv"])
u3=st.sidebar.file_uploader("Upload follow-up queue CSV",type=["csv"])
reports=normalize(pd.read_csv(u1) if u1 else pd.read_csv(DATA/"sample_anonymized_adr_reports.csv"))
exposure=normalize(pd.read_csv(u2) if u2 else pd.read_csv(DATA/"sample_medicine_exposure_context.csv"))
follow=normalize(pd.read_csv(u3) if u3 else pd.read_csv(DATA/"sample_safety_followup_queue.csv"))
required=["report_id","medicine_id","medicine_name","drug_class","reaction_term","seriousness","onset_days","report_completeness_pct","report_date"]
missing=[c for c in required if c not in reports.columns]
if missing: st.error("Missing ADR report columns: "+", ".join(missing)); st.stop()
reports=score_signals(reports); reports["report_date"]=pd.to_datetime(reports.report_date,errors="coerce")
med=aggregate(reports)
meds=["All"]+sorted(med.medicine_name.astype(str).unique()); bands=["All","Low","Moderate","High","Critical"]
sm=st.sidebar.selectbox("Medicine",meds); sb=st.sidebar.selectbox("Priority band",bands); threshold=st.sidebar.slider("Minimum review-priority score",0,100,0)
view=reports.copy()
if sm!="All": view=view[view.medicine_name==sm]
if sb!="All": view=view[view.medicine_id.isin(med.loc[med.priority_band.astype(str)==sb,"medicine_id"])]
med_view=aggregate(view); med_view=med_view[med_view.review_priority>=threshold]
if view.empty or med_view.empty: st.warning("No records match the current filters."); st.stop()

st.markdown("""<div class="hero"><div class="eyebrow">ADVERSE DRUG REACTION • LOCAL-FIRST • EXPLAINABLE SIGNAL SCREENING</div>
<h1>Prioritize pharmacovigilance signals for qualified human safety review.</h1>
<p>Organize anonymized safety reports, recurrence patterns, seriousness context, timing, and follow-up workload without making causality or treatment claims.</p>
<span class="pill">💊 Medicine Signals</span><span class="pill">📊 Reaction Patterns</span><span class="pill">🚨 Seriousness Review</span><span class="pill">📅 Trend Monitor</span><span class="pill">🧑‍⚕️ Human Follow-up</span><span class="pill">🔒 Local Processing</span></div>""",unsafe_allow_html=True)

k=[("ADR reports",len(view),"Filtered anonymized reports"),("High / Critical",int((med_view.review_priority>=60).sum()),"Medicine-level review signals"),
("Critical",int((med_view.review_priority>=80).sum()),"Highest review-priority band"),("Serious reports",int((view.seriousness!="Non-serious").sum()),"Seriousness context"),
("Avg completeness",f"{view.report_completeness_pct.mean()*100:.0f}%","Report completeness")]
cs=st.columns(5)
for c,(a,b,d) in zip(cs,k): c.markdown(f'<div class="card"><div class="label">{a}</div><div class="value">{b}</div><div class="sub">{d}</div></div>',unsafe_allow_html=True)

if page=="Dashboard":
    st.markdown('<div class="section">Safety-signal overview</div>',unsafe_allow_html=True)
    a,b,c=st.columns([1,1.25,1])
    with a:
        mix=med_view.priority_band.astype(str).value_counts().reindex(bands[1:]).fillna(0).reset_index(); mix.columns=["band","count"]
        st.plotly_chart(px.pie(mix,names="band",values="count",hole=.62,title="Medicine priority mix",template="plotly_white"),use_container_width=True)
    with b:
        st.plotly_chart(px.scatter(med_view,x="report_count",y="avg_signal",size="review_priority",color="review_priority",hover_name="medicine_name",text="medicine_name",
                                   color_continuous_scale=["#36aa73","#f0b34a","#e05d69"],title="Recurrence × signal strength",template="plotly_white"),use_container_width=True)
    with c:
        top=med_view.sort_values("review_priority",ascending=False).head(7)
        st.plotly_chart(px.bar(top.sort_values("review_priority"),x="review_priority",y="medicine_name",orientation="h",text_auto=".0f",range_x=[0,100],title="Top medicine review priorities",template="plotly_white"),use_container_width=True)
    d,e=st.columns(2)
    with d:
        weekly=view.set_index("report_date").resample("W")["signal_score"].mean().reset_index()
        st.plotly_chart(px.line(weekly,x="report_date",y="signal_score",markers=True,title="Weekly signal trend",template="plotly_white"),use_container_width=True)
    with e:
        rx=view.groupby("reaction_term",as_index=False).agg(reports=("report_id","count"),avg_signal=("signal_score","mean")).sort_values("reports",ascending=False).head(10)
        st.plotly_chart(px.bar(rx,x="reports",y="reaction_term",orientation="h",text_auto=True,title="Most reported reactions",template="plotly_white"),use_container_width=True)
    st.markdown('<div class="section">Medicine review queue</div>',unsafe_allow_html=True)
    st.dataframe(med_view.sort_values("review_priority",ascending=False).head(25),use_container_width=True,hide_index=True)
    if ASSET.exists():
        with st.expander("Dashboard visual reference"): st.image(str(ASSET),use_container_width=True)

elif page=="Medicine Signals":
    st.markdown('<div class="section">Medicine-level signals</div>',unsafe_allow_html=True)
    st.dataframe(med_view.sort_values("review_priority",ascending=False),use_container_width=True,hide_index=True)
    st.plotly_chart(px.scatter(med_view,x="serious_reports",y="report_count",size="review_priority",color="review_priority",hover_name="medicine_name",
                               color_continuous_scale=["#36aa73","#f0b34a","#e05d69"],title="Report recurrence × serious-report context",template="plotly_white"),use_container_width=True)

elif page=="Reaction Patterns":
    st.markdown('<div class="section">Reaction pattern analysis</div>',unsafe_allow_html=True)
    rp=view.groupby(["reaction_term","seriousness"],as_index=False).size(); rp.columns=["reaction_term","seriousness","reports"]
    st.plotly_chart(px.bar(rp,x="reaction_term",y="reports",color="seriousness",barmode="group",title="Reaction patterns by seriousness",template="plotly_white"),use_container_width=True)
    heat=view.pivot_table(index="medicine_name",columns="reaction_term",values="report_id",aggfunc="count",fill_value=0)
    st.plotly_chart(px.imshow(heat,text_auto=True,aspect="auto",title="Medicine × reaction matrix",color_continuous_scale=["#eef7f2","#7fcaa3","#2e7fd7"],template="plotly_white"),use_container_width=True)

elif page=="Seriousness Review":
    st.markdown('<div class="section">Seriousness and follow-up review</div>',unsafe_allow_html=True)
    ser=view.groupby("medicine_name",as_index=False).agg(reports=("report_id","count"),serious=("seriousness",lambda s:int((s!="Non-serious").sum())),hospitalization=("seriousness",lambda s:int((s=="Hospitalization").sum())))
    ser["serious_pct"]=(ser.serious/ser.reports*100).round(1)
    a,b=st.columns(2)
    with a: st.plotly_chart(px.bar(ser.sort_values("serious_pct"),x="serious_pct",y="medicine_name",orientation="h",text_auto=".1f",range_x=[0,100],title="Serious-report share",template="plotly_white"),use_container_width=True)
    with b: st.plotly_chart(px.scatter(ser,x="reports",y="serious_pct",size="hospitalization",color="serious_pct",hover_name="medicine_name",title="Volume × seriousness context",template="plotly_white"),use_container_width=True)
    st.dataframe(ser.sort_values("serious_pct",ascending=False),use_container_width=True,hide_index=True)

elif page=="Trend Monitor":
    st.markdown('<div class="section">Temporal signal monitoring</div>',unsafe_allow_html=True)
    trend=view.set_index("report_date").resample("W").agg(reports=("report_id","count"),avg_signal=("signal_score","mean")).reset_index()
    a,b=st.columns(2)
    with a: st.plotly_chart(px.line(trend,x="report_date",y="reports",markers=True,title="Weekly ADR report volume",template="plotly_white"),use_container_width=True)
    with b: st.plotly_chart(px.line(trend,x="report_date",y="avg_signal",markers=True,title="Weekly average screening signal",template="plotly_white"),use_container_width=True)
    st.dataframe(view.sort_values("report_date",ascending=False)[["report_id","medicine_name","reaction_term","seriousness","onset_days","signal_score","report_date"]].head(40),use_container_width=True,hide_index=True)

elif page=="Review Queue":
    st.markdown('<div class="section">Human safety-review queue</div>',unsafe_allow_html=True)
    fq=follow.copy(); fq["days_to_next_review"]=num(fq.get("days_to_next_review",pd.Series(index=fq.index)))
    merged=fq.merge(med[["medicine_id","medicine_name","review_priority","priority_band"]],on=["medicine_id","medicine_name"],how="left")
    statuses=["All"]+sorted(merged.review_status.astype(str).unique()); status=st.selectbox("Review status",statuses)
    if status!="All": merged=merged[merged.review_status==status]
    merged=merged.sort_values(["review_priority","days_to_next_review"],ascending=[False,True])
    st.dataframe(merged,use_container_width=True,hide_index=True)
    st.download_button("⬇️ Download review queue CSV",merged.to_csv(index=False).encode(),file_name="adr_human_review_queue.csv",mime="text/csv")

elif page=="Scenario Lab":
    st.markdown('<div class="section">Signal-review scenario lab</div>',unsafe_allow_html=True)
    st.caption("Controls change screening weights for review prioritization. They do not establish causality or alter clinical interpretation.")
    a,b,c,d=st.columns(4)
    with a: sw=st.slider("Seriousness weight",20,80,52,4)
    with b: rw=st.slider("Recurrence weight",10,50,27,3)
    with c: tw=st.slider("Timing weight",5,35,12,3)
    with d: cw=st.slider("Completeness weight",0,25,6,1)
    w=np.array([sw,rw,tw,cw],dtype=float); w=w/w.sum()
    sc=med_view.copy()
    sc["scenario_priority"]=(w[0]*(sc.serious_reports/sc.report_count.replace(0,np.nan)*100).fillna(0)+w[1]*sc.report_count.clip(0,100)+w[2]*(100-sc.median_onset_days.clip(0,30)/30*100)+w[3]*(100-sc.avg_completeness*100)).clip(0,100).round(1)
    sc["priority_shift"]=(sc.scenario_priority-sc.review_priority).round(1)
    st.dataframe(sc[["medicine_name","review_priority","scenario_priority","priority_shift","report_count","serious_reports"]].sort_values("scenario_priority",ascending=False),use_container_width=True,hide_index=True)
    st.download_button("⬇️ Download scenario CSV",sc.to_csv(index=False).encode(),file_name="adr_signal_scenario_results.csv",mime="text/csv")

elif page=="Reports & Export":
    st.markdown('<div class="section">Reports & export</div>',unsafe_allow_html=True)
    export=med_view.sort_values("review_priority",ascending=False); st.dataframe(export,use_container_width=True,hide_index=True)
    st.download_button("⬇️ Download medicine signals CSV",export.to_csv(index=False).encode(),file_name="adr_medicine_signal_summary.csv",mime="text/csv")
    st.download_button("⬇️ Download ADR reports CSV",view.to_csv(index=False).encode(),file_name="adr_reports_scored.csv",mime="text/csv")
    st.download_button("⬇️ Download exposure context CSV",exposure.to_csv(index=False).encode(),file_name="medicine_exposure_context.csv",mime="text/csv")

st.markdown("""<div class="warning"><b>Important:</b> This is pharmacovigilance signal-screening software. Signals do not establish that a medicine caused an adverse reaction, diagnose a patient, recommend treatment, determine product liability, or replace validated case review, clinical judgment, regulatory pharmacovigilance workflows, or applicable reporting requirements.</div>""",unsafe_allow_html=True)
st.markdown('<div class="footer">100% local CSV processing • No external APIs • Explainable heuristics • Human safety review required • Synthetic sample data</div>',unsafe_allow_html=True)
