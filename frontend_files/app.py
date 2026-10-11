import streamlit as st
import requests, pandas as pd, os

st.set_page_config(page_title="SuperKart Predictor", layout="wide")
st.title("SuperKart Sales Revenue Predictor")

default_backend = os.environ.get("BACKEND_URL", "http://backend:7860")
BACKEND_URL = st.sidebar.text_input("Backend URL", value=default_backend)

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        pw = st.number_input("Weight (kg)", 1.0, 25.0, 12.66, 0.1)
        ps = st.selectbox("Sugar", ["Low Sugar", "Regular", "No Sugar"])
        pa = st.number_input("Display Area", 0.0, 0.3, 0.027, format="%.3f")
        pm = st.number_input("MRP", 10.0, 300.0, 117.08, 0.5)
    with col2:
        ss = st.selectbox("Store Size", ["Small", "Medium", "High"])
        ct = st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
        stype = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
    with col3:
        pic = st.selectbox("Category Code", ["FD", "DR", "NC"])
        sa = st.number_input("Store Age", 1, 50, 16)
        ptc = st.selectbox("Type Category", ["Perishables", "Non Perishables"])
    if st.button("Predict", type="primary", use_container_width=True):
        payload = {"Product_Weight": pw, "Product_Sugar_Content": ps, "Product_Allocated_Area": pa,
            "Product_MRP": pm, "Store_Size": ss, "Store_Location_City_Type": ct, "Store_Type": stype,
            "Product_Id_char": pic, "Store_Age_Years": sa, "Product_Type_Category": ptc}
        try:
            r = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
            if r.status_code == 200:
                st.success(f"Predicted Sales: Rs {r.json()['prediction']:,.2f}")
            else:
                st.error(r.text)
        except Exception as e:
            st.error(str(e))
with tab2:
    uf = st.file_uploader("Upload CSV", type=["csv"])
    if uf:
        df_up = pd.read_csv(uf)
        st.dataframe(df_up.head(10))
        if st.button("Run Batch", type="primary", use_container_width=True):
            try:
                uf.seek(0)
                r = requests.post(f"{BACKEND_URL}/v1/predictbatch",
                                  files={"file": uf.getvalue()}, timeout=120)
                if r.status_code == 200:
                    preds = r.json()
                    st.success(f"{len(preds)} predictions!")
                    st.dataframe(pd.DataFrame({"Row": list(preds.keys()),
                                               "Predicted": list(preds.values())}))
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(str(e))
