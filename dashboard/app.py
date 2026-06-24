import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import sqlite3 
import io


# CONFIGURARE

st.set_page_config(
    page_title = "Sentiment Analysis Dashboard",
    page_icon = "🎬",
    layout = "wide",
    initial_sidebar_state = "expanded"
)


#API_URL = "http://127.0.0.1:8000"
API_URL = "https://madaluna83-sentiment-api.hf.space"
DB_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.db")
HISTORY_FILE = "dashboard/history.json"




# CSS personalizat 

st.html("""
<style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p  { margin: 0.3rem 0 0 0; opacity: 0.75; font-size: 0.95rem; }
    .result-card-pos {
        background: linear-gradient(135deg, #d5f5e3, #a9dfbf);
        border-left: 5px solid #27ae60;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-card-neg {
        background: linear-gradient(135deg, #fadbd8, #f1948a);
        border-left: 5px solid #e74c3c;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-label  { font-size: 1.6rem; font-weight: 700; margin: 0; }
    .result-conf   { font-size: 1rem; opacity: 0.8; margin: 0.2rem 0 0 0; }
    section[data-testid="stSidebar"] { background: #1a1a2e; }
    section[data-testid="stSidebar"] * { color: white !important; }
    footer { visibility: hidden; }
</style>
""")

#Database SQLITE

def init_db():
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                 id         INTEGER PRIMARY KEY AUTOINCREMENT,
                 timestamp  TEXT,
                 text       TEXT,
                 sentiment  TEXT,
                 confidence REAL
                 )
    """)
    conn.commit()
    conn.close()

def save_to_db(text, sentiment, confidence):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO analyses (timestamp, text, sentiment, confidence) VALUES (?, ?, ?, ?)",
        (datetime.now().strftime("%d.%m.%Y %H:%M"), text[:200], sentiment, confidence)
    )
    conn.commit()
    conn.close()

def load_from_db():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM analyses ORDER BY id DESC LIMIT 500", conn
    )
    conn.close()
    return df

def clear_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM analyses")
    conn.commit()
    conn.close()

init_db()

#Functii API

def call_api(text): 
    try: 
        r = requests.post(
            f"{API_URL}/predict", 
            json = {"text": text},
            timeout = 30
        )
        return r.json() if r.status_code == 200 else None
    except:
        return None

def call_api_batch(texts):
    try:
        payload = [{"text": t} for t in texts]
        r = requests.post(
            f"{API_URL}/predict/batch",
            json = payload, 
            timeout = 60
        )
        return r.json() if r.status_code == 200 else None
    except: 
        return None

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except:
        return False
    

# Highligh cuvinte

POSITIVE_WORDS = {
    "amazing", "great", "excellent", "love", "wonderful",
    "fantastic", "brilliant", "outstanding", "superb",
    "perfect", "beautiful", "enjoyed", "best", "awesome",
    "incredible", "magnificent", "delightful", "masterpiece"
}

NEGATIVE_WORDS = {
    "terrible", "awful", "boring", "waste", "bad",
    "horrible", "disappointing", "dreadful", "poor",
    "worst", "stupid", "ridiculous", "pathetic", "disgusting",
    "mediocre", "uninspired", "predictable", "tedious"
}

def highlight_words(text):
    words = text.split()
    result = []
    for word in words:
        clean = word.lower().strip(".,!?;:\"'")
        if clean in POSITIVE_WORDS:
            result.append(f"**:green[{word}]**")
        elif clean in NEGATIVE_WORDS:
            result.append(f"**:red[{word}]**")
        else:
            result.append(word)
    return " ".join(result)

#Sidebar pt navigare 

with st.sidebar:
    st.markdown(" Sentiment Analysis ")
    st.markdown(" ... ")

    page = st.radio(
        "Navigare",
        [
            "Analyze text",
            "Batch analysis",
            "Comparator",
            "Upload CSV",
            "Statistics",
            "History",
            "About"
        ],
        label_visibility="collapsed"
    )

    st.markdown(" ... ")

    #status api in sidebar

    api_ok = check_api()
    if api_ok:
        st.success("API online")
    else:
        st.error("API offline")

    st.markdown(" ... ")
    st.html("<small style='opacity:0.5'>DistilBERT - IMDb - 2026 </small>")


#Header principal 

st.html("""
<div class="main-header">
            <h1> Sentiment Analysis Dashboard </h1>
            <p> DistilBERT fine-tuned oe IMDb &nbsp;|&nbsp;
                Accuracy: <strong>92,8%</strong> &nbsp;|&nbsp;
                F1-Score: <strong>0.928</strong> &nbsp;|&nbsp;
                25.000 train exemples </p>
</div>
            """)

if not api_ok:
    st.error("Api offline - start 'uvicorn main:app --reload'")
    st.stop()

# PAGINA 1 - ANALIZA TEXT SIMPLU

if page == "Analyze text":
    col1, col2 = st.columns([1.1,0.9], gap="large")

    with col1:
        st.subheader("Enter text")

        text_input = st.text_area(
            "",
            placeholder="Ex: This movie was absolutely amazing,"
                        "I loved every second of it!",
            height=160,
            label_visibility="collapsed"
        )

        c1,c2 = st.columns(2)
        with c1:
            analyze = st.button("Analyze",
                                type="primary",
                                use_container_width=True)
        with c2:
            clear = st.button("Clear", 
                              use_container_width=True)
        
        if clear: 
            st.rerun()
        
        result = None
        if analyze and text_input.strip():
            with st.spinner("Analyze ..."):
                result = call_api(text_input)

            if result:
                sentiment = result["sentiment"]
                confidence = result["confidence"]

                card_class = "result-card-pos" if sentiment == "POSITIV" else "result-card-neg"
                
                st.html(f"""
                <div class="{card_class}">
                    <p class="result-label"> {sentiment}</p>
                    <p class="result-conf">
                        Scor de incredere: <strong>
                        {confidence*100:.1f}%</strong>
                    </p>
                </div>    
                            """)
                
                #Bara progres
                st.progress(confidence)

                #Interpretare
                if confidence >= 0.95:
                    st.info("The model is very confident about this prediction")
                elif confidence >= 0.80:
                    st.info("The model is confident about this prediction")
                else:
                    st.warning("Ambiguous text — the model is less certain")

                #Evidentiere cuvinte 
                st.markdown("**Key words identified:**")
                highlighted = highlight_words(text_input)
                st.markdown(
                    f"> {highlighted}"
                )

                st.caption(
                    "Green = positive words &nbsp;&nbsp;"
                    "Red = negative words"
                )

                save_to_db(text_input, sentiment, confidence)

            else: 
                st.error("Erorr connecting the API")

        elif analyze and not text_input.strip():
            st.warning("Enter a text for analysis: ")

    with col2: 
        st.subheader("Visual results: ")

        df_hist = load_from_db()
        if not df_hist.empty and analyze and text_input.strip() and result is not None:
            #Gauge chart
            conf_val = result["confidence"] * 100
            sent_color = "#27ae60" if result["sentiment"] == "POSITIV" else "#e74c3c"

            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta", 
                value=conf_val,
                number={"suffix": "%", "font": {"size": 40}},
                title={"text": result["sentiment"],
                       "font": {"size": 22}},
                delta={"reference": 50},
                gauge={
                    "axis": {"range": [0, 100],
                             "tickwidth": 1},
                    "bar":  {"color": sent_color},
                    "steps": [
                        {"range": [0, 50],  "color": "#e5c0f6"},
                        {"range": [50, 80], "color": "#fdebd0"},
                        {"range": [80, 100],"color": "#d5f5e3"}
                    ],
                    "threshold": {
                        "line":  {"color": "#333", "width": 3},
                        "thickness": 0.75,
                        "value": conf_val
                    }
                }
            ))

            fig_gauge.update_layout(
                height=280,
                margin=dict(t=60, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        else:
            st.html("""
            <div style="height:280px; display:flex;
                        align-items:center; justify-content:center;
                        background:#f8f9fa; border-radius:10px;
                        color:#aaa; font-size:1rem;">
                The gauge appears after the first analysis
            </div>
                        """)
            
        #Mini statistici 
        if not df_hist.empty:
            total = len(df_hist)
            n_poz = len(df_hist[df_hist["sentiment"] == "POZITIV"])
            n_neg = total - n_poz
            avg_conf = df_hist["confidence"].mean() * 100

            st.markdown(" **Session statistics:** ")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Total", total)
            mc2.metric("Pozitive", n_poz)
            mc3.metric("Negative", n_neg)
            st.caption(f"Avg. confidence: {avg_conf:.1f}%")

# PAGINA 2 - ANALIZA BATCH 

elif page == "Batch analysis":
    st.subheader("Multiple analysis")
    st.markdown("Enter multiple texts, **one per line**.")

    batch_input = st.text_area(
        "", 
        placeholder = "This movie was great!\nTerrible film, waste of time.\n"
                    "It was okay, nothing special.",
        height=200,
        label_visibility="collapsed"
    )

    if st.button("Analyze all", type="primary",
                 use_container_width=True):
        lines = [l.strip() for l in batch_input.split("\n")
                 if l.strip()]
        if lines:
            with st.spinner(f"Analyze {len(lines)} texts..."):
                results_raw = call_api_batch(lines)

            if results_raw:
                rows = []
                for r in results_raw:
                    rows.append({
                        "Text":        r["text"][:80] + ("..." if len(r["text"]) > 80 else ""),
                        "Sentiment":   f"{r['sentiment']}",
                        "Incredere":   f"{r['confidence']*100:.1f}%",
                        "_sentiment":  r["sentiment"],
                        "_confidence": r["confidence"] 
                    })
                    save_to_db(r["text"], r["sentiment"], r["confidence"])

                df_batch = pd.DataFrame(rows)

                #  Metrici batch 

                n_poz = sum(1 for r in rows if r["_sentiment"] == "POZITIV")
                n_neg = len(rows) - n_poz 

                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Totally analyzed", len(rows))
                b2.metric("Pozitive", n_poz)
                b3.metric("Negative", n_neg)
                avg = sum(r["_confidence"] for r in rows) / len(rows)
                b4.metric("Avg. confidence", f"{avg*100:.1f}%")

                st.divider()

                # TAbel rezultate 
                st.dataframe(
                    df_batch[["Text", "Sentiment", "Incredere"]],
                    use_container_width=True
                )

                # Grafic pie rapid 
                fig = px.pie(
                    values=[n_poz, n_neg],
                    names=["Pozitiv", "Negativ"],
                    color_discrete_sequence=["#efe470", "#b93ce7"],
                    title="Distribution of batch feelings"
                )
                st.plotly_chart(fig, use_container_width=True)

                # EXport 
                export_df = df_batch[["Text", "Sentiment", "Incredere"]]
                csv = export_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV results", 
                    csv,
                    f"batch_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv",
                    use_container_width=True
                )
        else: 
            st.warning("Enter at least one text.")

# PAGINA 3 - COMPARATOR 
elif page == "Comparator": 
    st.subheader("Compare two texts")
    st.markdown("Analyze and compare the sentiment of two texts simultaneously.")

    col_a, col_b= st.columns(2, gap="large")

    with col_a:
        st.html("**Text A**")
        text_a = st.text_area("", height=140,
                              placeholder = "First text: ",
                              key = "comp_a", 
                              label_visibility = "collapsed")
        
    with col_b:
        st.html("**Text B**")
        text_b = st.text_area("", height=140,
                              placeholder = "Second text: ",
                              key = "comp_b", 
                              label_visibility = "collapsed")
        
    if st.button("Compare", type = "primary", use_container_width=True):
        if text_a.strip() and text_b.strip():
            with st.spinner("Analyzed.."):
                res_a = call_api(text_a)
                res_b = call_api(text_b)

            if res_a and res_b:
                st.divider()
                ca, cb = st.columns(2, gap="large")

                for col, res, text, label in [
                    (ca, res_a, text_a, "A"),
                    (cb, res_b, text_b, "B")
                ]:
                    with col: 
                        sent = res["sentiment"]
                        conf = res["confidence"]

                        if sent == "POZITIV":
                            st.success(f"### Text {label}: {sent}")
                        else:
                            st.error(f"### Text {label}: {sent}")

                        st.metric("Confidence", f"{conf*100:.1f}%")

                        # color = "#27ae60" if sent == "POSITIVE" else "#e74c3c"
                        # card = "result-card-pos" if sent == "POSITIVE" else "result-card-neg"

                        # st.html(f"""
                        # <div class="{card}">
                        #     <p class="result-label">
                        #         Text {label}: {sent}
                        #     <\p>
                        #     <p class="result-conf">
                        #         Incredere: <strong>{conf*100:.1f}%<\strong>
                        #     <\p>
                        # <\div>
                        #             """)
                        
                        st.progress(conf)
                        st.markdown(highlight_words(text))
                        save_to_db(text, sent, conf)

                # Grafic comparatie
                st.divider()
                fig_comp = go.Figure(data=[
                    go.Bar(
                        name = "Text A",
                        x=["Pozitiv", "Negativ"],
                        y=[
                            res_a["confidence"] if res_a["sentiment"] == "POZITIV"
                            else 1 - res_a["confidence"],
                            res_a["confidence"] if res_a["sentiment"] == "NEGATIV"
                            else 1 - res_a["confidence"]
                        ], 
                        marker_color = "#2b0ae9"
                    ), 
                    go.Bar(
                        name = "Text B",
                        x=["Pozitiv", "Negativ"],
                        y=[
                            res_b["confidence"] if res_b["sentiment"] == "POZITIV"
                            else 1 - res_b["confidence"],
                            res_b["confidence"] if res_b["sentiment"] == "NEGATIV"
                            else 1 - res_b["confidence"]
                        ], 
                        marker_color = "#e16e0a"
                    )
                ])

                fig_comp.update_layout(
                    barmode = "group", 
                    title = "Comparison of confidence scores", 
                    yaxis_tickformat = ".0%", 
                    height = 300
                )
                st.plotly_chart(fig_comp, use_container_width = True)
            else: 
                st.warning("Enter both texts for comparison!")

# PAGINA 4 - UPLOAD CSV

elif page == "Upload CSV":
    st.subheader("Analyze CSV file")
    st.markdown(
        "Load a CSV file with a column named 'text'."
        "Every row wil be analyzed automatically."
    )

    uploaded = st.file_uploader("Choose file:", type="csv")

    if uploaded:
        df_up = pd.read_csv(uploaded, sep=",", quotechar='"', on_bad_lines='skip')
        st.success(f"File upload: {len(df_up)} randuri")
        st.dataframe(df_up.head(5), use_container_width=True)

        if "text" not in df_up.columns:
            st.error("The file does not have a 'text' column")
        else: 
            if st.button("Analyze file",
                         type = "primary", use_container_width=True):
                results = []
                bar = st.progress(0)
                status = st.empty()

                for i, row in df_up.iterrows():
                    text = str(row["text"])
                    r = call_api(text)
                    if r:
                        results.append({
                            "text":       text,
                            "sentiment":  r["sentiment"],
                            "confidence": round(r["confidence"] * 100, 1)
                        })
                        save_to_db(text, r["sentiment"], r["confidence"])

                    bar.progress((i + 1) / len(df_up))
                    status.text(f"Analyze the row {i+1}/{len(df_up)}...")

                status.empty()
                bar.empty()

                if results:
                    df_out = pd.DataFrame(results)

                    # Statistici
                    n_poz = len(df_out[df_out["sentiment"] == "POZITIV"])
                    n_neg = len(df_out) - n_poz
                    u1, u2, u3 = st.columns(3)
                    u1.metric("Total", len(df_out))
                    u2.metric("Pozitive", n_poz)
                    u3.metric("Negative", n_neg)

                    st.divider()
                    st.dataframe(df_out, use_container_width=True)

                    # Grafice

                    fc1, fc2 = st.columns(2)

                    with fc1:
                        fig_p = px.pie(
                            values=[n_poz, n_neg],
                            names=["Pozitiv", "Negativ"],
                            color_discrete_sequence=["#27ae60", "#e74c3c"],
                            title="Distribution of feelings"
                        )
                        st.plotly_chart(fig_p, use_container_width=True)

                    with fc2:
                        fig_h = px.histogram(
                            df_out,
                            x="confidence",
                            color="sentiment",
                            nbins=20,
                            color_discrete_map={
                                "POZITIV": "#27ae60",
                                "NEGATIV": "#e74c3c"
                            },
                            title="Distribution of confidence scores"
                        )
                        st.plotly_chart(fig_h, use_container_width=True)

                    # Export
                    csv_out = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download the CSV results",
                        csv_out,
                        f"rezultate_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        "text/csv",
                        use_container_width=True
                    )

# PAGINA 5 - STATISTICI

elif page == "Statistics":
    st.subheader("Global statistics")

    df_all = load_from_db()

    if df_all.empty:
        st.info("No analysis performed yet."
                "Go to - Text Analysis - to get started.")
    else: 
        # Metrici principale
        total    = len(df_all)
        n_poz    = len(df_all[df_all["sentiment"] == "POZITIV"])
        n_neg    = total - n_poz
        avg_conf = df_all["confidence"].mean() * 100
        max_conf = df_all["confidence"].max() * 100
        min_conf = df_all["confidence"].min() * 100

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total analyses", total)
        s2.metric("Pozitive", n_poz,
                  f"{n_poz/total*100:.0f}%")
        s3.metric("Negative", n_neg,
                  f"{n_neg/total*100:.0f}%")
        s4.metric("Avg. confidence", f"{avg_conf:.1f}%")
 
        st.divider() 

        # Grafice 2x2
        g1, g2 = st.columns(2)

        with g1:
            # Pie chart
            fig_pie = px.pie(
                values=[n_poz, n_neg],
                names=["Pozitiv", "Negativ"],
                color_discrete_sequence=["#27ae60", "#e74c3c"],
                title="The distribution of feelings",
                hole=0.4
            )
            fig_pie.update_traces(
                textposition="inside",
                textinfo="percent+label"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with g2:
            # Histogram incredere
            fig_hist = px.histogram(
                df_all,
                x="confidence",
                color="sentiment",
                nbins=25,
                color_discrete_map={
                    "POZITIV": "#27ae60",
                    "NEGATIV": "#e74c3c"
                },
                title="Distribution of confidence scores",
                labels={
                    "confidence": "Confidence score",
                    "count": "Number of analyses"
                }
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        g3, g4 = st.columns(2)
 
        with g3:
            # Evolutie in timp
            df_all["id_order"] = range(len(df_all) - 1, -1, -1)
            fig_line = px.scatter(
                df_all,
                x="id_order",
                y="confidence",
                color="sentiment",
                color_discrete_map={
                    "POZITIV": "#27ae60",
                    "NEGATIV": "#e74c3c"
                },
                title="The evolution of confidence scores",
                labels={
                    "id_order": "Analyze #",
                    "confidence": "Score"
                },
            )
            st.plotly_chart(fig_line, use_container_width=True)
 
        with g4:
            # Box plot
            fig_box = px.box(
                df_all,
                x="sentiment",
                y="confidence",
                color="sentiment",
                color_discrete_map={
                    "POZITIV": "#27ae60",
                    "NEGATIV": "#e74c3c"
                },
                title="Distribution of scores by class",
                labels={
                    "sentiment":  "Sentiment",
                    "confidence": "Confidence score"
                }
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # Statistici detaliate
        st.divider()
        st.html("Detailed statistics per class:")
        stats = df_all.groupby("sentiment")["confidence"].agg([
            ("Medie", "mean"),
            ("Minim", "min"),
            ("Maxim", "max"),
            ("Std. Dev.", "std")
        ]).round(4) * 100
        st.dataframe(stats, use_container_width=True)


# PAGINA 6 - ISTORIC

elif page == "History":
    st.subheader("Analysis history")

    df_hist = load_from_db()
 
    if df_hist.empty:
        st.info("History is empty — analytics appear here automatically.")
    else:
        # Filtre
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            filter_sent = st.selectbox(
                "Filter feeling:",
                ["Toate", "POZITIV", "NEGATIV"]
            )
        with fc2:
            filter_conf = st.slider(
                "Minimal confidence (%):", 0, 100, 0
            )
        with fc3:
            st.metric("Filtered results",
                      len(df_hist))
 
        # Aplica filtre
        df_filtered = df_hist.copy()
        if filter_sent != "All:":
            df_filtered = df_filtered[
                df_filtered["sentiment"] == filter_sent
            ]
        df_filtered = df_filtered[
            df_filtered["confidence"] >= filter_conf / 100
        ]
 
        st.dataframe(
            df_filtered[["timestamp", "text",
                         "sentiment", "confidence"]].rename(columns={
                "timestamp":  "Data/Ora",
                "text":       "Text",
                "sentiment":  "Sentiment",
                "confidence": "Scor"
            }),
            use_container_width=True,
            height=400
        )

         # Export si stergere
        e1, e2 = st.columns(2)
        with e1:
            csv_hist = df_filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Export history (CSV)",
                csv_hist,
                f"istoric_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                use_container_width=True
            )
        with e2:
            if st.button("Delete history",
                         use_container_width=True,
                         type="secondary"):
                clear_db()
                st.success("History cleared!")
                st.rerun()

# PAGINA 7 - DESPRE MODEL 

elif page == "About":
    
    st.subheader("About the system")

    c1, c2 = st.columns(2, gap = "large")

    with c1: 
        st.markdown("The Model")
        st.markdown("""
        | Proprietate | Valoare |
        |---|---|
        | Arhitectură | DistilBERT |
        | Model de bază | distilbert-base-uncased |
        | Dataset antrenament | IMDb (25.000 exemple) |
        | Epoci antrenament | 3 (early stopping) |
        | Batch size | 32 |
        | Learning rate | 2e-5 (scheduler liniar) |
        | Max tokens | 512 |
        | Parametri | ~66 milioane |
        """)

        st.markdown("Performanta")
        st.markdown("""
        | Metrică | Valoare |
        |---|---|
        | Acuratețe | **92.8%** |
        | F1-Score | **0.928** |
        | Precision Negativ | 0.93 |
        | Recall Negativ | 0.93 |
        | Precision Pozitiv | 0.93 |
        | Recall Pozitiv | 0.93 |
        """)

    with c2: 
        st.markdown("Arhitectura sistemului")
        st.markdown("""
            ```
            Utilizator
                ↓
            Dashboard (Streamlit :8501)
                ↓  HTTP POST /predict
            API REST (FastAPI :8000)
                ↓
            Model DistilBERT
                ↓
            Sentiment + Confidence
            ```
        """)

        st.markdown("Tehnologii utilizate")
        st.markdown("""
        - **Python 3.11** — limbaj de programare
        - **PyTorch 2.x** — framework deep learning
        - **HuggingFace Transformers** — modele NLP
        - **FastAPI** — framework API REST
        - **Streamlit** — framework dashboard
        - **Plotly** — vizualizări interactive
        - **SQLite** — stocare istoric analize
        """)
            
        st.markdown("Referinte")
        st.markdown("""
        - Devlin et al. (2019) — *BERT: Pre-training of Deep
          Bidirectional Transformers*
        - Sanh et al. (2019) — *DistilBERT, a distilled version
          of BERT*
        - Maas et al. (2011) — *Learning Word Vectors for
          Sentiment Analysis (IMDb)*
                    """)
        
    # GrAfice comparatie arhitecturi 
    st.divider()
    st.html("Comparatie v1 vs v2")

    fig_comp = go.Figure(data=[
        go.Bar(
            name="v1 — Baseline",
            x=["Acuratete (%)", "F1-Score (x100)"],
            y=[89.0, 89.0],
            marker_color="#d3346b"
        ),
        go.Bar(
            name="v2 — Optimizat",
            x=["Acuratete (%)", "F1-Score (x100)"],
            y=[92.8, 92.8],
            marker_color="#23e0d7"
        )
    ])
    fig_comp.update_layout(
        barmode="group",
        title="Performanta: Baseline vs. Optimizat",
        yaxis_range=[80, 100],
        height=350
    )
    st.plotly_chart(fig_comp, use_container_width=True)


# Footer

st.html("""
<div style='text-align:center; color:#aaa;
            font-size:0.8em; margin-top:2rem;
            padding-top:1rem; border-top:1px solid #eee;'>
    Sentiment Analysis Dashboard &nbsp;•&nbsp;
    DistilBERT fine-tuned pe IMDb &nbsp;•&nbsp;
    Lucrare de disertație 2026
</div>
""")

