# 🏡 California Housing Price Predictor

### 🔮 Project Overview

This project is a **web-based application** that predicts the **median house value** in California based on various socio-economic and geographical features. It uses a **Random Forest Regressor**, which outperformed other models like **Decision Tree** and **Linear Regression**, trained on the **California Housing Dataset**. Users can enter housing details to get instant predictions.

The app also provides a **feature importance visualization**, helping users understand which factors most influence housing prices.

---

### 🧰 Key Features

- **Interactive User Input:** Enter housing details such as:
  - Longitude & Latitude
  - Median income
  - House age, total rooms, total bedrooms, population, households
  - Ocean proximity (categorical)
- **Real-time Prediction:**The app preprocesses your input using the **same pipeline as training data** and predicts the median house value with the Random Forest model.
- **Feature Importance Visualization:**Displays the top 10 features that most influence house price predictions using a colorful bar chart.
- **Smooth User Experience:**
  - Loading animations during prediction to show the processing of data.
  - Easy-to-read results in a clean, dark-themed interface.

---

### ⚙️ How It Works

1. **Data Collection:** Uses the California Housing Dataset and generates a synthetic `ocean_proximity` feature.
2. **Data Preprocessing:**
   - Numeric values are imputed (missing values replaced) and scaled.
   - Categorical values are imputed and one-hot encoded.
   - Ensures new input data is transformed exactly like the training data using a **pipeline**.
3. **Model Training & Selection:**
   - Tested multiple algorithms including **Linear Regression** and **Decision Tree Regressor**.
   - **Random Forest Regressor** gave the best performance and stability.
   - Ensemble of decision trees averages predictions for higher accuracy.
4. **Prediction:**
   - User inputs are transformed via the pipeline.
   - Random Forest predicts the median house value.
5. **Feature Explanation:**
   - The model calculates feature importance.
   - Top 10 features are visualized for better understanding.

---

### 🖥️ Technologies Used

- Python
- Streamlit (for the web interface)
- Scikit-learn (data preprocessing, pipeline, Random Forest, Decision Tree, Linear Regression)
- Pandas & NumPy (data manipulation)
- Plotly (feature importance visualization)

---

### ⚡ Author

**Rutvik Prajapati** – Data Science Enthusiast

Made with ❤️ using Python and Streamlit.
