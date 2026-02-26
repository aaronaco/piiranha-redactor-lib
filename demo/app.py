import streamlit as st
import pandas as pd
from piiranha_redactor import PIIRedactor

st.set_page_config(
    page_title="Piiranha Redactor",
    page_icon=":material/shield:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

SAMPLE_TEXTS = {
    "(select a sample)": "",
    "Personal introduction": (
        "Hi, my name is Priya Raghavan and I recently moved to 48 Westbourne Grove, London. "
        "You can reach me at priya.raghavan@protonmail.com or call +44 7911 123456. "
        "My date of birth is 14/03/1992."
    ),
    "Customer support email": (
        "Dear Support,\n\n"
        "I am writing regarding my account. My name is Carlos Mendoza, "
        "account number 4829-1038-7745. I noticed an unauthorized charge on my "
        "credit card ending in 8834 (full number: 4532 8921 0076 8834). "
        "Please contact me at carlos.mendoza@gmail.com or +34 612 345 678.\n\n"
        "My billing address is Calle de Serrano 27, 28001 Madrid.\n\n"
        "Regards,\nCarlos"
    ),
    "Medical referral note": (
        "Patient: Margaret O'Brien, DOB: 22/08/1965, SSN: 284-19-7234. "
        "Referred by Dr. Takeshi Yamamoto for cardiac evaluation. "
        "Patient resides at 1523 Maple Drive, Portland, OR 97205. "
        "Emergency contact: kevin.obrien@outlook.com, phone: (503) 555-0147."
    ),
    "Multi-language text": (
        "Je m'appelle Sophie Laurent, mon email est sophie.laurent@orange.fr. "
        "Mijn naam is Willem de Vries, woonachtig op Keizersgracht 412, Amsterdam. "
        "Mi nombre es Ana Garcia, telefono +34 654 321 987."
    ),
}


@st.cache_resource(show_spinner="Loading model (this may take a moment on first run)...")
def load_redactor():
    return PIIRedactor()


redactor = load_redactor()

st.title("Piiranha Redactor")
st.markdown("Local PII detection and redaction powered by DeBERTa-v3. All processing happens on-device.")

col_sample, col_threshold = st.columns([2, 1])
with col_sample:
    sample_key = st.selectbox("Load sample text", options=list(SAMPLE_TEXTS.keys()))
with col_threshold:
    threshold = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for an entity to be detected. Lower values catch more potential PII but may increase false positives.",
    )

initial_text = SAMPLE_TEXTS[sample_key] if sample_key != "(select a sample)" else ""

text_input = st.text_area(
    "Input text",
    value=initial_text,
    height=180,
    placeholder="Paste or type text containing personally identifiable information...",
)

run = st.button("Run analysis", type="primary")

if run and text_input.strip():
    result = redactor.redact_with_details(text_input, threshold=threshold)
    entities = result["entities"]
    redacted_text = result["redacted_text"]

    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Entities found", len(entities))
    col2.metric("Unique labels", len({e.label for e in entities}))
    if entities:
        avg_score = sum(e.score for e in entities) / len(entities)
        col3.metric("Avg. confidence", f"{avg_score:.2%}")
    else:
        col3.metric("Avg. confidence", "--")

    tab_redacted, tab_entities = st.tabs(["Redacted output", "Detected entities"])

    with tab_redacted:
        st.text_area("Redacted text", value=redacted_text, height=180, disabled=True)

    with tab_entities:
        if entities:
            df = pd.DataFrame([
                {
                    "Word": e.word,
                    "Label": e.label,
                    "Score": round(e.score, 4),
                    "Start": e.start,
                    "End": e.end,
                }
                for e in entities
            ])
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.info("No PII entities detected at the current threshold.")

    st.caption(f"Device: {redactor.device} | Model: piiranha-v1-detect-personal-information")

elif run:
    st.warning("Enter some text to analyze.")
