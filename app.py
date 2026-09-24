import streamlit as st
import pandas as pd
import joblib


# ==========================================
# 1. Load Model
# ==========================================

model = joblib.load("purchase_model.pkl")


# ==========================================
# 2. Page Configuration
# ==========================================

st.set_page_config(
    page_title="Retail Purchase Predictor",
    page_icon="🛒",
    layout="wide"
)


# ==========================================
# 3. Header
# ==========================================

st.title("🛒 Retail Customer Purchase Prediction")

st.markdown(
    """
    **AI-powered prediction of whether an online shopping session
    is likely to result in a purchase.**
    """
)

st.info(
    "Enter the customer's browsing and session information below "
    "and click **Predict Purchase**."
)


# ==========================================
# 4. Browsing Behaviour
# ==========================================

st.subheader("📊 Browsing Behaviour")

col1, col2, col3 = st.columns(3)

with col1:

    administrative = st.number_input(
        "Administrative Pages",
        min_value=0,
        value=0,
        step=1
    )

    administrative_duration = st.number_input(
        "Administrative Duration",
        min_value=0.0,
        value=0.0
    )

    informational = st.number_input(
        "Informational Pages",
        min_value=0,
        value=0,
        step=1
    )

    informational_duration = st.number_input(
        "Informational Duration",
        min_value=0.0,
        value=0.0
    )


with col2:

    product_related = st.number_input(
        "Product Related Pages",
        min_value=0,
        value=1,
        step=1
    )

    product_related_duration = st.number_input(
        "Product Related Duration",
        min_value=0.0,
        value=0.0
    )

    page_values = st.number_input(
        "Page Value",
        min_value=0.0,
        value=0.0
    )


with col3:

    bounce_rates = st.number_input(
        "Bounce Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.02,
        format="%.4f"
    )

    exit_rates = st.number_input(
        "Exit Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.04,
        format="%.4f"
    )

    special_day = st.number_input(
        "Special Day",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        format="%.2f"
    )


# ==========================================
# 5. Visitor Information
# ==========================================

st.subheader("👤 Visitor Information")

col4, col5, col6 = st.columns(3)

with col4:

    month = st.selectbox(
        "Month",
        [
            "Feb",
            "Mar",
            "May",
            "June",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec"
        ]
    )


with col5:

    visitor_type = st.selectbox(
        "Visitor Type",
        [
            "Returning_Visitor",
            "New_Visitor",
            "Other"
        ]
    )


with col6:

    weekend = st.selectbox(
        "Weekend",
        [False, True],
        format_func=lambda x: "Yes" if x else "No"
    )


# ==========================================
# 6. Technical Session Information
# ==========================================

st.subheader("💻 Session Information")

col7, col8, col9, col10 = st.columns(4)

with col7:

    operating_systems = st.number_input(
        "Operating System",
        min_value=1,
        value=2,
        step=1
    )

with col8:

    browser = st.number_input(
        "Browser",
        min_value=1,
        value=2,
        step=1
    )

with col9:

    region = st.number_input(
        "Region",
        min_value=1,
        value=1,
        step=1
    )

with col10:

    traffic_type = st.number_input(
        "Traffic Type",
        min_value=1,
        value=2,
        step=1
    )


# ==========================================
# 7. Prediction
# ==========================================

st.divider()

predict_button = st.button(
    "🔮 Predict Purchase",
    use_container_width=True
)


if predict_button:

    # Create input dataframe
    input_data = pd.DataFrame({

        "Administrative": [administrative],

        "Administrative_Duration": [
            administrative_duration
        ],

        "Informational": [informational],

        "Informational_Duration": [
            informational_duration
        ],

        "ProductRelated": [product_related],

        "ProductRelated_Duration": [
            product_related_duration
        ],

        "BounceRates": [bounce_rates],

        "ExitRates": [exit_rates],

        "PageValues": [page_values],

        "SpecialDay": [special_day],

        "Month": [month],

        "OperatingSystems": [
            operating_systems
        ],

        "Browser": [browser],

        "Region": [region],

        "TrafficType": [traffic_type],

        "VisitorType": [visitor_type],

        "Weekend": [weekend]
    })


    # Prediction
    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]


    # Find probability of purchase class
    purchase_index = list(model.classes_).index(True)

    purchase_probability = probabilities[
        purchase_index
    ]


    # ======================================
    # Result
    # ======================================

    st.subheader("🎯 Prediction Result")

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        if prediction:

            st.success(
                "🛒 Prediction: Purchase"
            )

        else:

            st.warning(
                "🛍️ Prediction: No Purchase"
            )


    with result_col2:

        st.metric(
            "Purchase Probability",
            f"{purchase_probability * 100:.2f}%"
        )


    # ======================================
    # Probability Breakdown
    # ======================================

    st.write("### Prediction Probability Breakdown")

    no_purchase_probability = 1 - purchase_probability

    probability_data = pd.DataFrame(
        {
            "Outcome": [
                "No Purchase",
                "Purchase"
            ],

            "Probability (%)": [
                no_purchase_probability * 100,
                purchase_probability * 100
            ]
        }
    )

    st.bar_chart(
        probability_data.set_index("Outcome")
    )


# ==========================================
# 8. Project Information
# ==========================================

st.divider()

with st.expander("ℹ️ About this Project"):

    st.write(
        """
        **Problem:** Retail Customer Purchase Prediction

        **Dataset:** Online Shoppers Purchasing Intention Dataset

        **Target:** Revenue

        **Machine Learning Model:** Random Forest Classifier

        **Evaluation:** Accuracy, Precision, Recall and F1 Score

        **Interface:** Streamlit
        """
    )