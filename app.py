import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------
# Page Configuration
# ------------------------------

st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    page_icon="🏨",
    layout="wide"
)

# ------------------------------
# Custom CSS
# ------------------------------

st.markdown("""
<style>

.main{
    background:#F8F9FA;
}

h1{
    color:#003566;
}

.metric-container{
    background:#ffffff;
    padding:15px;
    border-radius:12px;
    box-shadow:2px 2px 10px lightgray;
}

.sidebar .sidebar-content{
    background:#001D3D;
}

</style>
""",unsafe_allow_html=True)

# ------------------------------
# Load Dataset
# ------------------------------

df = pd.read_csv("hotel_bookings.csv")

# ------------------------------
# Load Model
# ------------------------------

model = joblib.load("data.pkl.gz")

# ------------------------------
# Sidebar
# ------------------------------

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/139/139899.png",
    width=100
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "",
    [
        "🏠 Home",
        "📊 Dashboard",
        "📂 Dataset",
        "🤖 Prediction",
        "📈 Model",
        "ℹ About"
    ]
)

# ------------------------------
# HOME PAGE
# ------------------------------

if page=="🏠 Home":

    st.title("🏨 Hotel Booking Cancellation Prediction")

    st.write("")

    st.image(
        "https://images.unsplash.com/photo-1566073771259-6a8506099945",
        use_container_width=True
    )

    st.write("")

    st.markdown("""
## Project Overview

This application predicts whether a hotel booking is likely to be cancelled.

### Technologies Used

- Python
- Pandas
- Scikit-Learn
- Streamlit
- Plotly
- Machine Learning

### Algorithm

✔ Logistic Regression

✔ Random Forest Classifier
""")

    st.write("---")

    st.subheader("Project Features")

    col1,col2,col3=st.columns(3)

    with col1:
        st.success("📊 Interactive Dashboard")

    with col2:
        st.success("🤖 ML Prediction")

    with col3:
        st.success("📈 Business Insights")

# ------------------------------
# DASHBOARD
# ------------------------------

elif page=="📊 Dashboard":

    st.title("📊 Dashboard")

    total=len(df)

    cancelled=df["is_canceled"].sum()

    avg_adr=round(df["adr"].mean(),2)

    avg_stay=round(
        (
            df["stays_in_week_nights"]+
            df["stays_in_weekend_nights"]
        ).mean(),2
    )

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Total Bookings",total)

    c2.metric("Cancelled",cancelled)

    c3.metric("Average ADR",avg_adr)

    c4.metric("Average Stay",avg_stay)

    st.write("---")

    fig=px.pie(
        df,
        names="is_canceled",
        title="Booking Cancellation"
    )

    st.plotly_chart(fig,use_container_width=True)

    fig=px.histogram(
        df,
        x="lead_time",
        color="is_canceled",
        title="Lead Time Distribution"
    )

    st.plotly_chart(fig,use_container_width=True)

    # ------------------------------
# DATASET PAGE
# ------------------------------

elif page=="📂 Dataset":

    st.title("📂 Hotel Booking Dataset")

    st.write("### Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.write("### Dataset Shape")
    col1, col2 = st.columns(2)

    col1.info(f"Rows : {df.shape[0]}")
    col2.info(f"Columns : {df.shape[1]}")

    st.write("### Data Types")
    st.dataframe(df.dtypes.astype(str))

    st.write("### Missing Values")
    st.dataframe(df.isnull().sum())


# ------------------------------
# STATISTICS PAGE
# ------------------------------

elif page=="📈 Statistics":

    st.title("📈 Statistical Analysis")

    st.subheader("Dataset Statistics")

    st.dataframe(df.describe())

    st.subheader("Correlation Matrix")

    corr = df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Viridis",
        title="Correlation Heatmap"
    )

    st.plotly_chart(fig, use_container_width=True)


# ------------------------------
# VISUALIZATION PAGE
# ------------------------------

elif page=="📊 Visualization":

    st.title("📊 Data Visualization Dashboard")

    # --------------------
    # Cancellation Count
    # --------------------

    fig = px.bar(
        df["is_canceled"].value_counts().reset_index(),
        x="is_canceled",
        y="count",
        color="is_canceled",
        title="Booking Cancellation Count"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------
    # Hotel Type
    # --------------------

    fig = px.pie(
        df,
        names="hotel",
        title="Hotel Type Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------
    # Lead Time
    # --------------------

    fig = px.histogram(
        df,
        x="lead_time",
        nbins=40,
        color="is_canceled",
        title="Lead Time Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------
    # ADR Distribution
    # --------------------

    fig = px.box(
        df,
        y="adr",
        color="is_canceled",
        title="Average Daily Rate"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------
    # Adults vs ADR
    # --------------------

    fig = px.scatter(
        df,
        x="adults",
        y="adr",
        color="is_canceled",
        title="Adults vs ADR"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------
    # Monthly Bookings
    # --------------------

    fig = px.histogram(
        df,
        x="arrival_date_month",
        color="is_canceled",
        title="Bookings by Month"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------
    # Customer Type
    # --------------------

    fig = px.bar(
        df["customer_type"].value_counts().reset_index(),
        x="customer_type",
        y="count",
        color="customer_type",
        title="Customer Type"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
# PREDICTION PAGE
# ------------------------------

elif page == "🤖 Prediction":

    st.title("🤖 Hotel Booking Cancellation Prediction")

    st.write("Enter the booking details below.")

    col1, col2 = st.columns(2)

    hotel = col1.selectbox("Hotel", [0,1])

    lead_time = col2.number_input("Lead Time",0,800,50)

    arrival_year = col1.number_input("Arrival Year",2015,2030,2017)

    arrival_month = col2.number_input("Arrival Month (Encoded)",0,11,5)

    arrival_week = col1.number_input("Arrival Week",1,53,20)

    arrival_day = col2.number_input("Arrival Day",1,31,15)

    weekend = col1.number_input("Weekend Nights",0,10,1)

    week = col2.number_input("Week Nights",0,20,2)

    adults = col1.number_input("Adults",1,10,2)

    children = col2.number_input("Children",0,10,0)

    babies = col1.number_input("Babies",0,5,0)

    meal = col2.number_input("Meal (Encoded)",0,10,0)

    country = col1.number_input("Country (Encoded)",0,300,50)

    market = col2.number_input("Market Segment",0,10,3)

    channel = col1.number_input("Distribution Channel",0,10,1)

    repeated = col2.number_input("Repeated Guest",0,1,0)

    previous_cancel = col1.number_input("Previous Cancellations",0,20,0)

    previous_booking = col2.number_input("Previous Bookings",0,100,0)

    reserved = col1.number_input("Reserved Room",0,20,1)

    assigned = col2.number_input("Assigned Room",0,20,1)

    booking_changes = col1.number_input("Booking Changes",0,20,0)

    deposit = col2.number_input("Deposit Type",0,5,0)

    agent = col1.number_input("Agent",0,600,0)

    waiting = col2.number_input("Waiting List Days",0,500,0)

    customer = col1.number_input("Customer Type",0,5,0)

    adr = col2.number_input("ADR",0.0,1000.0,100.0)

    parking = col1.number_input("Parking Spaces",0,10,0)

    special = col2.number_input("Special Requests",0,10,0)


    if st.button("Predict"):

        data = [[
            hotel,
            lead_time,
            arrival_year,
            arrival_month,
            arrival_week,
            arrival_day,
            weekend,
            week,
            adults,
            children,
            babies,
            meal,
            country,
            market,
            channel,
            repeated,
            previous_cancel,
            previous_booking,
            reserved,
            assigned,
            booking_changes,
            deposit,
            agent,
            waiting,
            customer,
            adr,
            parking,
            special,
            
        ]]

        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error("❌ Booking will be Cancelled")
        else:
            st.success("✅ Booking will NOT be Cancelled")

elif page=="📈 Model":

    st.title("📈 Machine Learning Model")

    st.success("Model Used : Random Forest Classifier")

    st.write("### Model Information")

    st.write("""
- Classification Problem
- Supervised Machine Learning
- Target Variable : is_canceled
- Train-Test Split : 80:20
- Evaluation Metrics:
    - Accuracy
    - Confusion Matrix
    - Classification Report
    """)

    st.info("Random Forest generally performs better than Logistic Regression for this dataset.")
elif page=="ℹ About":
    st.title("ℹ About")

    st.markdown("""
## Hotel Booking Cancellation Prediction

### Objective

Predict whether a hotel booking will be cancelled using Machine Learning.

### Dataset

Hotel Booking Demand Dataset

### Algorithms

- Logistic Regression
- Random Forest Classifier

### Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Plotly
- Streamlit



<center>

Made with ❤️ using Streamlit

Hotel Booking Cancellation Prediction

</center>
""",
unsafe_allow_html=True
)