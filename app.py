import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="AI Business Insight System",
    page_icon="📊",
    layout="wide"
)

st.title("AI Business Insight System")
st.subheader("Prototyp systemu wspierającego decyzje przedsiębiorstwa")

st.write(
    """
    Aplikacja będzie służyła do analizy danych sprzedażowych oraz tekstowych.
    Docelowo system będzie prezentował wskaźniki sprzedażowe, analizę opinii klientów,
    porównanie metod klasycznych i AI oraz rekomendacje decyzyjne.
    """
)

st.info("Jeżeli widzisz ten komunikat, aplikacja Streamlit działa poprawnie.")

sample_data = pd.DataFrame({
    "Miesiąc": ["Styczeń", "Luty", "Marzec", "Kwiecień"],
    "Sprzedaż": [12000, 15000, 13500, 18000]
})

fig = px.line(
    sample_data,
    x="Miesiąc",
    y="Sprzedaż",
    markers=True,
    title="Testowy wykres sprzedaży"
)

st.plotly_chart(fig, width="stretch")
st.dataframe(sample_data)