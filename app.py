import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ---------------------------------------------------
# Load saved model, scaler and feature list
# ---------------------------------------------------

kmeans_model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
cluster_features = joblib.load("cluster_features.pkl")


# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Student Profile Clustering",
    page_icon="🎓",
    layout="wide"
)


# ---------------------------------------------------
# App Title
# ---------------------------------------------------

st.title("🎓 Students' Social Network Profile Clustering")

st.write(
    """
    This application uses **K-Means Clustering** to segment students
    based on their social network profile, number of friends,
    age, and interests.
    """
)


# ---------------------------------------------------
# Basic Information
# ---------------------------------------------------

st.header("Student Profile")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=13.0,
        max_value=30.0,
        value=17.0,
        step=0.1
    )

with col2:
    number_of_friends = st.number_input(
        "Number of Friends",
        min_value=0,
        value=20,
        step=1
    )


# ---------------------------------------------------
# Interest Inputs
# ---------------------------------------------------

st.header("Student Interests")

st.write(
    "Enter the frequency/count associated with each interest "
    "in the student's social network profile."
)

# Everything other than age and NumberOffriends
# comes directly from the feature list saved from your notebook.

interest_cols = [
    feature
    for feature in cluster_features
    if feature not in ["age", "NumberOffriends"]
]

interest_values = {}

# Display interests in 3 columns
columns = st.columns(3)

for i, feature in enumerate(interest_cols):

    with columns[i % 3]:

        interest_values[feature] = st.number_input(
            feature.capitalize(),
            min_value=0,
            value=0,
            step=1,
            key=feature
        )


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

st.markdown("---")

if st.button("Find Student Segment", type="primary"):

    # Create input dictionary
    input_data = {
        "age": age,

        # Same transformation used in the notebook
        "NumberOffriends": np.log1p(number_of_friends)
    }

    # Add all interest values
    input_data.update(interest_values)

    # Convert to dataframe
    input_df = pd.DataFrame([input_data])

    # IMPORTANT:
    # Keep exactly the same feature order used during training
    input_df = input_df[cluster_features]

    # Apply the saved StandardScaler
    input_scaled = scaler.transform(input_df)

    # Predict K-Means cluster
    cluster = kmeans_model.predict(input_scaled)[0]


    # ---------------------------------------------------
    # Display Result
    # ---------------------------------------------------

    st.success(f"Student belongs to Cluster {cluster}")

    if cluster == 0:

        st.subheader("Highly Engaged / Interest-Expressive Student")

        st.write(
            """
            This student belongs to the segment showing relatively
            higher activity across multiple interests.

            Based on the cluster profiling performed in the project,
            this group showed stronger frequencies for interests such
            as **hair, music, dance, god, cute, shopping, mall,
            band and rock**.
            """
        )

    elif cluster == 1:

        st.subheader("Lower-Engagement / General-Interest Student")

        st.write(
            """
            This student belongs to the larger segment characterized
            by comparatively lower frequencies across most recorded
            interests.

            **Music, god, dance, shopping and band** were among the
            more common interests within this segment.
            """
        )


# ---------------------------------------------------
# Model Information
# ---------------------------------------------------

st.markdown("---")

st.header("About the Model")

st.write(
    """
    Three clustering algorithms were evaluated in this project:

    - **K-Means**
    - **Hierarchical Clustering**
    - **DBSCAN**

    Although DBSCAN achieved the highest silhouette score, it produced
    highly imbalanced clusters and classified several observations as
    noise.

    Therefore, **K-Means with 2 clusters** was selected as the final
    model for student segmentation because it provided more meaningful
    and interpretable student groups.
    """
)