from __future__ import annotations

import time
from collections import deque

import joblib
import pandas as pd
import streamlit as st
import altair as alt

from ids.config import ARTIFACTS_DIR, RAW_DIR, DatasetConfig

st.set_page_config(page_title="IDS Live Demo (CICIDS2017)", layout="wide")


@st.cache_resource
def load_model():
    model_path = ARTIFACTS_DIR / "ids_model.joblib"
    meta_path = ARTIFACTS_DIR / "metadata.joblib"
    if not model_path.exists() or not meta_path.exists():
        st.error("Model not found. Train first: python -m ids.models.train")
        st.stop()
    pipe = joblib.load(model_path)
    meta = joblib.load(meta_path)
    return pipe, meta


@st.cache_data
def load_data(nrows: int = 20000, shuffle: bool = True):
    cfg = DatasetConfig()
    csv_path = RAW_DIR / cfg.raw_filename
    if not csv_path.exists():
        st.error(f"Dataset not found at {csv_path}")
        st.stop()

    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()

    if shuffle:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    if nrows is not None:
        df = df.head(nrows)

    return df


def is_benign_label(label: str) -> bool:
    label = str(label).strip().lower()
    return label in {"normal traffic", "benign", "normal"}


def make_score_chart(stream_df: pd.DataFrame, threshold: float):
    base = alt.Chart(stream_df).encode(
        x=alt.X("t:Q", title="Stream index"),
    )

    line = base.mark_line().encode(
        y=alt.Y("score:Q", title="Attack probability", scale=alt.Scale(domain=[0, 1]))
    )

    points = base.mark_circle(size=55, opacity=0.9).encode(
        y="score:Q",
        color=alt.Color(
            "pred_label:N",
            scale=alt.Scale(
                domain=["benign", "attack"],
                range=["#2ecc71", "#e74c3c"],
            ),
            title="Prediction",
        ),
        tooltip=["t:Q", "score:Q", "pred_label:N", "true_label:N"],
    )

    rule = (
        alt.Chart(pd.DataFrame({"threshold": [threshold]}))
        .mark_rule(color="#f1c40f", strokeDash=[6, 6])
        .encode(y="threshold:Q")
    )

    return (line + points + rule).properties(height=360).interactive()


def main():
    st.title("Live Intrusion Detection System Demo")
    st.caption(
        "Streams CICIDS2017 rows like network flows and shows real-time ML predictions."
    )

    pipe, meta = load_model()
    label_col = str(meta["label_col"]).strip()

    with st.sidebar:
        st.header("Controls")
        nrows = st.slider("Rows to stream", 2000, 50000, 15000, step=1000)
        speed = st.slider("Stream speed (rows/sec)", 1, 50, 10)
        threshold = st.slider("Attack threshold", 0.01, 0.99, 0.50, step=0.01)
        shuffle = st.checkbox("Shuffle traffic", value=True)
        show_raw = st.checkbox("Show raw row features", value=False)
        start = st.button("Start / Restart")

    df = load_data(nrows=nrows, shuffle=shuffle)

    colA, colB = st.columns([2, 1])

    with colA:
        st.subheader("Live Score Stream")
        chart_placeholder = st.empty()
        table_placeholder = st.empty()

    with colB:
        st.subheader("Alerts")
        alert_placeholder = st.empty()
        total_placeholder = st.empty()

    if not start:
        st.info("Use the sidebar and click **Start / Restart** to begin streaming.")
        return

    # Stream buffers
    window = 300
    scores = deque(maxlen=window)
    preds = deque(maxlen=window)
    timestamps = deque(maxlen=window)
    true_labels = deque(maxlen=window)

    alerts = []
    total_streamed = 0

    for i in range(len(df)):
        row = df.iloc[i : i + 1].copy()

        y_true_label = None
        if label_col in row.columns:
            y_true_label = str(row[label_col].iloc[0])
            row = row.drop(columns=[label_col])

        if hasattr(pipe, "predict_proba"):
            score = float(pipe.predict_proba(row)[:, 1][0])
        else:
            score = float(pipe.decision_function(row)[0])

        pred_label = "attack" if score >= threshold else "benign"

        scores.append(score)
        preds.append(pred_label)
        timestamps.append(i)
        true_labels.append(y_true_label)

        total_streamed += 1

        if pred_label == "attack":
            alerts.append(
                {
                    "index": i,
                    "score": round(score, 4),
                    "pred": "attack",
                    "true_label": y_true_label,
                }
            )

        stream_df = pd.DataFrame(
            {
                "t": list(timestamps),
                "score": list(scores),
                "pred_label": list(preds),
                "true_label": list(true_labels),
            }
        )

        chart_placeholder.altair_chart(
            make_score_chart(stream_df, threshold),
            use_container_width=True,
        )

        recent = stream_df.tail(20)
        table_placeholder.dataframe(
            recent[["t", "score", "pred_label", "true_label"]],
            use_container_width=True,
        )

        alert_placeholder.dataframe(
            pd.DataFrame(alerts[-25:]), use_container_width=True
        )
        total_placeholder.metric("Total streamed", total_streamed)

        if show_raw:
            st.write("Raw features (current row):")
            st.dataframe(row, use_container_width=True)

        time.sleep(1.0 / max(speed, 1))

    st.success("Stream finished.")


if __name__ == "__main__":
    main()
