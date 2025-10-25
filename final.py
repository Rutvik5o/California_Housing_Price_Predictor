import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px


st.set_page_config(
    page_title="California Housing Price Predictor 🏠",
    page_icon="🏡",
    layout="wide",
)


st.markdown("""
<style>
.stApp { background-color: #0B0C10; color: #C5C6C7; font-family: 'Segoe UI', sans-serif; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1F2833 0%, #0B0C10 100%); color: #66FCF1; }
            
h1, h2, h3, h4 { color: #45A29E; text-shadow: 1px 1px 2px black; }
            
div.stButton > button { background-color: #45A29E; color: #0B0C10; font-weight: 700; border-radius: 12px; font-size: 16px; transition: transform 0.3s, background 0.3s; }
div.stButton > button:hover { transform: scale(1.05); background-color: #66FCF1; }
            
.stDataFrame, .dataframe { background-color: #1F2833 !important; color: #C5C6C7 !important; border-radius: 10px; }
            
</style>
""", unsafe_allow_html=True)


st.title("🏡 California Housing Price Predictor")
st.markdown("### 🔮 Predict   house values using a trained Random Forest Model")
st.markdown("---")


MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"


def train_model():
    st.info("📦 Loading California Housing dataset...")
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    df = df.rename(columns={
        "MedInc": "median_income",
        "HouseAge": "housing_median_age",
        "AveRooms": "total_rooms",
        "AveBedrms": "total_bedrooms",
        "Population": "population",
        "AveOccup": "households",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "MedHouseVal": "median_house_value"
    })
    df["ocean_proximity"] = np.random.choice(["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"], size=len(df))


    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    numeric_features = ["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income"]
    categorical_features = ["ocean_proximity"]

    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")), ('scaler', StandardScaler())])

    cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy="most_frequent")), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([('num', num_pipeline, numeric_features), ('cat', cat_pipeline, categorical_features)])

    pipeline = Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(n_estimators=100, random_state=42))])

    with st.spinner("🚀 Training model..."):

        progress = st.progress(0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        for i in range(0, 101, 5):
            time.sleep(0.05)
            progress.progress(i)
        pipeline.fit(X_train, y_train)

    joblib.dump(pipeline.named_steps['model'], MODEL_FILE)

    joblib.dump(preprocessor, PIPELINE_FILE)
    st.success("✅ Model trained and saved successfully!")


if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):

    st.warning("⚠️ Model or pipeline not found! Train first.")

    if st.button("🛠 Train Model Now"):
        train_model()
    st.stop()
-
model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)

-
st.sidebar.header("🏠 Enter Housing Details")

longitude = st.sidebar.number_input("Longitude", value=-121.89, step=0.1)

latitude = st.sidebar.number_input("Latitude", value=37.29, step=0.1)

housing_median_age = st.sidebar.number_input("Housing Median Age", value=30, min_value=1)

total_rooms = st.sidebar.number_input("Total Rooms", value=880, min_value=1)

total_bedrooms = st.sidebar.number_input("Total Bedrooms", value=129, min_value=1)

population = st.sidebar.number_input("Population", value=322, min_value=1)

households = st.sidebar.number_input("Households", value=126, min_value=1)

median_income = st.sidebar.number_input("Median Income (10k USD)", value=8.3252, min_value=0.0, step=0.1)

ocean_proximity = st.sidebar.selectbox("Ocean Proximity 🌊", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])


input_data = pd.DataFrame({
    "longitude": [longitude],
    "latitude": [latitude],
    "housing_median_age": [housing_median_age],
    "total_rooms": [total_rooms],
    "total_bedrooms": [total_bedrooms],
    "population": [population],
    "households": [households],
    "median_income": [median_income],
    "ocean_proximity": [ocean_proximity]
})


tab1, tab2, tab3 = st.tabs(["📝 Input Data", "🚀 Prediction", "🎯 Feature Importance"])


with tab1:
    st.subheader("📋 Input Data Preview")
    st.dataframe(input_data, width='stretch')

with tab2:
    if st.button("Predict   House Value"):
        # Loading animation
        with st.spinner("🔮 Calculating prediction..."):
            progress = st.progress(0)
            for i in range(0, 101, 5):
                time.sleep(0.03)
                progress.progress(i)
            transformed_input = pipeline.transform(input_data)
            prediction = model.predict(transformed_input)[0]

        st.markdown(f"""
        <div style="background-color:#1F2833; padding:20px; border-radius:15px; color:#66FCF1; font-size:24px; text-align:center;">
            🏠 Predicted   House Value: <b>${prediction:,.2f}</b>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    if st.button("Show Top 10 Features"):
        with st.spinner("📊 Fetching feature importance..."):
            progress = st.progress(0)
            for i in range(0, 101, 5):
                time.sleep(0.03)
                progress.progress(i)

            try:
                num_features = input_data.drop("ocean_proximity", axis=1).columns.tolist()

                cat_features = list(pipeline.named_transformers_["cat"]["onehot"].get_feature_names_out(["ocean_proximity"]))

                all_features = num_features + cat_features

                importances = model.feature_importances_
                feat_imp = pd.DataFrame({"Feature": all_features, "Importance": importances}).sort_values("Importance", ascending=True).tail(10)

                fig = px.bar(feat_imp, x="Importance",
                              y="Feature", orientation='h',
                             text="Importance", color="Importance", 
                             color_continuous_scale="Turbo", 
                             height=400)
                

                fig.update_layout(plot_bgcolor="#0B0C10", paper_bgcolor="#0B0C10", font_color="#C5C6C7") #fixing layout

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("⚠️ Could not compute feature importances.") #catching interrupted value
                st.text(str(e))


with st.expander("📘 About This App"):
    st.markdown("""
    This application uses a **Random Forest Regressor** trained on the California Housing Dataset  
    to predict the **  house value** based on geographical & socio-economic features.
    """)

with st.expander("🧠 Why Use Pipelines"):
    st.markdown("""
    Pipelines ensure **new input data** is preprocessed exactly as during training, including:
    - Handling missing values
    - Feature scaling
    - One-hot encoding categorical variables
    """)

with st.expander("🚧 Future Improvements"):
    st.markdown("""
    - 🌎 Add map-based visualization  
    - 🧮 Integrate SHAP or LIME for explainability
    """)

st.markdown("""
<div style="font-size:18px; text-align:center; color:#C5C6C7;">
⚡ Made with by <b>Rutvik Prajapati</b> | Data Science Enthusiast
</div>
""", unsafe_allow_html=True)
