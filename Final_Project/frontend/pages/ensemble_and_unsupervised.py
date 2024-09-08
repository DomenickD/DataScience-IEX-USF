"""The more advanced machine learning techniques."""

import streamlit as st

st.title("Explore Machine Learning Techniques")

st.markdown(
    "### Welcome! Let's explore some fascinating techniques in \
    machine learning through simple explanations, interactive \
        examples, and visual aids."
)

st.divider()

# Introduction with simplified explanation
st.header("What Are Ensemble and Unsupervised Techniques?")
st.markdown(
    """
In the world of machine learning, there are different techniques \
    used to make predictions and uncover patterns in data. \
        Today, we're exploring two key concepts:
- **Ensemble Techniques:** Think of it like a group of \
    friends working together to solve a problem. By combining \
        their strengths, they can often find better solutions \
            than any one of them could alone.
- **Unsupervised Techniques:** Imagine trying to organize \
    your photos into groups without any labels. Unsupervised \
        techniques help find patterns in data without \
            knowing the answers in advance.
"""
)

st.divider()

# Section for Ensemble Techniques with interactive elements
st.header("Ensemble Techniques: Working Together for Better Results")
st.markdown(
    """
Ensemble techniques combine multiple models to improve accuracy \
    and make more reliable predictions. Let's look at some \
        popular ensemble methods:
"""
)

# Expander for Bagging
with st.expander("📦 Bagging (Bootstrap Aggregating)"):
    st.markdown(
        """
   **Bagging** is like asking several experts for their opinion \
    and then taking an average. 
   - **Example:** Random Forest
   - **How It Works:** Multiple models are trained on different \
    random samples of the data. Their predictions are averaged to make the final decision.
   """
    )
    st.image("Pictures/random_forest.png")
    st.caption(
        "Source: \
            https://corporatefinanceinstitute.com/resources/data-science/random-forest/"
    )

# Expander for Boosting
with st.expander("🚀 Boosting"):
    st.markdown(
        """
   **Boosting** is about learning from mistakes. 
   - **Example:** AdaBoost, Gradient Boosting
   - **How It Works:** Each model in the sequence \
    focuses on the mistakes made by the previous ones, making \
        the overall model stronger.
   """
    )
    st.image("Pictures/adaboost.png")
    st.caption("Source: https://datamapu.com/posts/classical_ml/adaboost/")

# Expander for Stacking
with st.expander("🔗 Stacking"):
    st.markdown(
        """
   **Stacking** is like getting a final opinion from a top \
    expert who combines all other opinions.
   - **How It Works:** Different types of models are combined, \
    and a final model (meta-learner) makes the final prediction \
        based on all previous models.
   """
    )

st.divider()

# Section for Unsupervised Techniques with interactive elements
st.header("Unsupervised Techniques: Finding Hidden Patterns")
st.markdown(
    """
Unsupervised learning helps us discover patterns and structures \
    in data without having labeled outcomes. Let's explore \
        some common methods:
"""
)

# Expander for Clustering
with st.expander("🔍 Clustering"):
    st.markdown(
        """
   **Clustering** is like grouping similar items together.
   - **Example:** K-Means, Hierarchical Clustering
   - **How It Works:** Data points are grouped based on similarity. \
    This is useful in customer segmentation, image compression, etc.
   """
    )
    st.image("Pictures/kmeans.gif")
    st.caption("Source: https://giphy.com/explore/kmeans")


# Expander for Dimensionality Reduction
with st.expander("📉 Dimensionality Reduction"):
    st.markdown(
        """
   **Dimensionality Reduction** helps to simplify data by \
    reducing the number of features.
   - **Example:** PCA, t-SNE
   - **How It Works:** It condenses data into fewer dimensions \
    while preserving important information. Great for visualization and speeding up algorithms.
   """
    )

# Expander for Anomaly Detection
with st.expander("🚨 Anomaly Detection"):
    st.markdown(
        """
   **Anomaly Detection** is like finding the needle in a \
    haystack.
   - **Example:** Isolation Forest, One-Class SVM
   - **How It Works:** These techniques identify unusual \
    patterns that don't fit with the rest of the data, \
        often used for fraud detection.
   """
    )

st.divider()
