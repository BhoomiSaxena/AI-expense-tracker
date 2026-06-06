import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_data, clean_data, add_category, get_summary, category_summary, is_income_category

st.title("💰 AI Expense Tracker")

uploaded_file = st.file_uploader("Upload your bank CSV", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    df = clean_data(df)
    df = add_category(df)
    df.to_csv("processed_data.csv", index=False)

```
    st.subheader("📄 Raw Data")
    st.write(df)

income, expense = get_summary(df)

st.metric("Total Income", f"₹{income}")
st.metric("Total Expense", f"₹{expense}")

st.subheader("📊 Category-wise Spending")
cat_data = category_summary(df)

fig = px.pie(
    values=cat_data.values,
    names=cat_data.index,
    title="Spending Distribution"
)
st.plotly_chart(fig)

st.subheader("📈 Spending Over Time")

df['Month'] = df['Date'].dt.strftime('%Y-%m')

income_mask = is_income_category(df['Category'])

monthly = (
    df.loc[~income_mask]
    .assign(Spend=lambda x: x['Amount'].abs())
    .groupby('Month')['Spend']
    .sum()
    .reset_index()
)

fig2 = px.line(
    monthly,
    x='Month',
    y='Spend',
    title="Monthly Trend"
)

st.plotly_chart(fig2)

st.subheader("💡 Smart Advice")

if expense > income:
    st.warning("You are spending more than you earn!")
elif expense > 0.7 * income:
    st.info("Try reducing your expenses to save more.")
else:
    st.success("Great! You are managing your finances well.")
```
