import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, mean_squared_error

# Load dataset
df = pd.read_excel("transaction.xlsx", engine="openpyxl", dtype=str)

st.title("Tourism Recommendation System Evaluation")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ----- MAP Calculation -----
if "Rating" in df.columns:
    df["relevant"] = (df["Rating"] >= 3).astype(int)
    df["score"] = df["Rating"] / df["Rating"].max()

    map_score = average_precision_score(df["relevant"], df["score"])
    st.write("Mean Average Precision (MAP):", map_score)

# ----- RMSE Calculation -----
if "Rating" in df.columns:
    y_test = df["Rating"]
    y_pred = df["Rating"].mean() * np.ones(len(df))

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write("Root Mean Squared Error (RMSE):", rmse)


st.title("Tourism Recommendation System")

# Load dataset
df = pd.read_excel("transaction.xlsx")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------- Visit Mode Prediction Comment --------
st.subheader("Visit Mode Analysis")

if "VisitMode" in df.columns:
    
    # Mode count analysis
    mode_counts = df["VisitMode"].value_counts()
    st.write("Visit Mode Distribution:")
    st.bar_chart(mode_counts)

    # Comment based on data
    most_common_mode = df["VisitMode"].mode()[0]

    st.write(" Prediction Insight:")
    st.success(f"Most visitors prefer *{most_common_mode}* visit mode.")

    st.write("""
    Comment:
    This shows visitor behavior pattern based on historical transaction data.
    Businesses can use this insight for tourism planning.
    """)

# -------- MAP Calculation --------
if "Rating" in df.columns:
    df["relevant"] = (df["Rating"] >= 3).astype(int)
    df["score"] = df["Rating"] / df["Rating"].max()

    map_score = average_precision_score(df["relevant"], df["score"])
    st.write("MAP Score:", map_score)

# -------- RMSE Calculation --------
if "Rating" in df.columns:
    y_test = df["Rating"]
    y_pred = df["Rating"].mean() * np.ones(len(df))

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write("RMSE Score:", rmse)

    st.subheader("Recommended Attractions")

# Check required columns
if "UserID" in df.columns and "AttractionID" in df.columns and "Rating" in df.columns:

    # User profile based recommendation (Simple Logic)
    user_avg_rating = df.groupby("UserID")["Rating"].mean().reset_index()
    user_avg_rating.columns = ["UserID", "UserPreferenceScore"]

    st.write("User Profile Preference:")
    st.dataframe(user_avg_rating.head())

    # Attraction recommendation based on rating popularity
    attraction_score = df.groupby("AttractionID")["Rating"].mean().sort_values(ascending=False)

    st.write("Top Recommended Attractions:")
    st.dataframe(attraction_score.head(10))

    # Comment
    st.success("Recommendations generated based on user transaction history and rating behavior.")

else:
    st.warning("Required columns (UserID, AttractionID, Rating) not found.")

st.subheader("Tourism Data Visualization")

# -------- Popular Attractions --------
st.write("### Popular Attractions")

if "AttractionID" in df.columns:
    top_attractions = df["AttractionID"].value_counts().head(10)
    st.bar_chart(top_attractions)

    st.write("Most Popular Attractions are shown based on visit frequency.")

# -------- Top Regions Visualization --------
st.write("### Top Regions")

if "Region" in df.columns:
    top_regions = df["Region"].value_counts().head(10)
    st.bar_chart(top_regions)

    st.write("Top tourist regions based on transaction history.")

# -------- User Segmentation --------
st.write("### User Segmentation")

if "UserID" in df.columns:

    user_visits = df["UserID"].value_counts()

    # Segment users based on visit frequency
    def segment_user(x):
        if x >= 10:
            return "Frequent Visitor"
        elif x >= 5:
            return "Regular Visitor"
        else:
            return "Occasional Visitor"

    user_segment = user_visits.apply(segment_user)

    st.write("User Segment Distribution:")
    st.bar_chart(user_segment.value_counts())

    st.write("User segmentation based on visit behavior.")