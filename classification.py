import streamlit as st
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
@st.cache_data
def load_data():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    df['target'] = df['target'].apply(lambda x: 1 if x == 1 else 0)  # Convert to binary: 1 = class_1, 0 = others
    return df, wine.feature_names

df, feature_names = load_data()

# Train-test split
X = df.iloc[:, :-1]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)



# UI
st.title("🍷 Wine Quality Classifier")
st.write("Adjust the sliders to input wine chemical properties and predict quality.")

# Sidebar sliders
st.sidebar.title("Input Features")

inputs = []
for col in feature_names:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    value = st.sidebar.slider(col, min_val, max_val, float(df[col].mean()))
    inputs.append(value)

input_data = [inputs]

# Prediction
prediction = model.predict(input_data)[0]
result = "High Quality (Class 1)" if prediction == 1 else "Low Quality (Other)"

st.subheader("Prediction Result")
st.write(f"🔍 Predicted wine quality is: **{result}**")