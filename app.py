import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay

st.set_page_config(page_title="Customer Behaviour Analyzer", page_icon="🛍️", layout="wide")
st.title("Customer Behaviour Analysis & Segmentation")
st.caption("Machine learning dashboard • Online Retail dataset")

@st.cache_data
def load_data():
    customers = pd.read_csv("customer_features.csv")
    model_results = pd.read_csv("model_comparison.csv")
    purchases = pd.read_csv("cleaned_purchases.csv", parse_dates=["InvoiceDate"])
    customers["CustomerID"] = customers["CustomerID"].astype(int)
    purchases["CustomerID"] = purchases["CustomerID"].astype(int)
    return customers, model_results, purchases

try:
    customers, model_results, purchases = load_data()
except Exception as e:
    st.error("Place customer_features.csv, model_comparison.csv, and cleaned_purchases.csv in the same folder as app.py.")
    st.exception(e)
    st.stop()

features = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "TotalQuantity", "UniqueProducts"]
page = st.sidebar.radio("Navigate", ["Overview", "Customer Segmentation", "Purchase Prediction", "Model Comparison"])

if page == "Overview":
    st.header("Business Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Identifiable customers", f'{customers["CustomerID"].nunique():,}')
    c2.metric("Valid purchase lines", f"{len(purchases):,}")
    c3.metric("Distinct invoices", f'{purchases["InvoiceNo"].nunique():,}')
    c4.metric("Recorded revenue", f'£{purchases["Revenue"].sum():,.0f}')
    st.info("Revenue is based on positive-quantity purchase lines after the project's initial cleaning; cancellations and non-positive quantities/prices are excluded.")
    left, right = st.columns(2)
    with left:
        st.subheader("Customers by segment")
        fig, ax = plt.subplots()
        customers["Cluster"].value_counts().sort_index().plot(kind="bar", ax=ax)
        ax.set_xlabel("K-Means cluster"); ax.set_ylabel("Customers")
        st.pyplot(fig); plt.close(fig)
    with right:
        st.subheader("Top countries by revenue")
        vals = purchases.groupby("Country")["Revenue"].sum().nlargest(10).sort_values()
        fig, ax = plt.subplots()
        vals.plot(kind="barh", ax=ax); ax.set_xlabel("Revenue (£)")
        st.pyplot(fig); plt.close(fig)
    st.subheader("Customer profiles")
    st.dataframe(customers.head(25), use_container_width=True)

elif page == "Customer Segmentation":
    st.header("Customer Segmentation")
    clusters = sorted(customers["Cluster"].dropna().unique().tolist())
    selected = st.multiselect("Show clusters", clusters, default=clusters)
    X = customers[features].replace([np.inf, -np.inf], np.nan)
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    Xlog = np.log1p(np.clip(Xi, 0, None))
    Xscaled = StandardScaler().fit_transform(Xlog)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xscaled)
    plot = customers[["CustomerID", "Cluster"]].copy()
    plot["PC1"], plot["PC2"] = coords[:, 0], coords[:, 1]
    fig, ax = plt.subplots(figsize=(9, 5))
    for cl, group in plot[plot["Cluster"].isin(selected)].groupby("Cluster"):
        ax.scatter(group["PC1"], group["PC2"], s=16, alpha=.6, label=f"Cluster {cl}")
    ax.set_xlabel("Principal Component 1"); ax.set_ylabel("Principal Component 2")
    ax.set_title("Customer segments projected using PCA"); ax.legend()
    st.pyplot(fig); plt.close(fig)
    st.caption(f"Two components explain {pca.explained_variance_ratio_.sum()*100:.1f}% of variance. PCA is used for visualization.")
    view = customers[customers["Cluster"].isin(selected)]
    profile = view.groupby("Cluster").agg(
        CustomerCount=("CustomerID", "nunique"), AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"), AvgSpending=("Monetary", "mean"),
        MedianSpending=("Monetary", "median"), AvgOrderValue=("AvgOrderValue", "mean"),
        AvgQuantity=("TotalQuantity", "mean"), AvgUniqueProducts=("UniqueProducts", "mean")
    ).round(2)
    st.subheader("Segment profiles"); st.dataframe(profile, use_container_width=True)
    if len(view):
        cid = st.number_input("Search Customer ID", min_value=0, step=1, value=int(view["CustomerID"].iloc[0]))
        found = customers[customers["CustomerID"] == cid]
        if not found.empty: st.dataframe(found, use_container_width=True)
        else: st.warning("Customer ID not found.")

elif page == "Purchase Prediction":
    st.header("Future Purchase Prediction")
    st.write("Features use transactions through 30 September 2011. The target is whether a customer purchases from 1 October through 29 December 2011.")
    cutoff = pd.Timestamp("2011-09-30 23:59:59")
    end = pd.Timestamp("2011-12-29 23:59:59")
    hist = purchases[purchases["InvoiceDate"] <= cutoff].copy()
    future = purchases[(purchases["InvoiceDate"] > cutoff) & (purchases["InvoiceDate"] <= end)].copy()
    if hist.empty or future.empty:
        st.error("Expected observation/prediction dates are not present in cleaned_purchases.csv."); st.stop()
    ref = cutoff.normalize() + pd.Timedelta(days=1)
    hx = hist.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (ref - x.max()).days),
        Frequency=("InvoiceNo", "nunique"), Monetary=("Revenue", "sum"),
        TotalQuantity=("Quantity", "sum"), UniqueProducts=("StockCode", "nunique"))
    hx["AvgOrderValue"] = hx["Monetary"] / hx["Frequency"]
    future_ids = set(future["CustomerID"].unique())
    y = pd.Series(hx.index.isin(future_ids).astype(int), index=hx.index)
    X = hx[features].replace([np.inf, -np.inf], np.nan)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    choice = st.selectbox("Classifier", ["Logistic Regression", "Decision Tree", "SVM", "Neural Network", "Naive Bayes"])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, early_stopping=True, random_state=42),
        "Naive Bayes": GaussianNB()
    }
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", models[choice])])
    pipe.fit(Xtr, ytr); pred = pipe.predict(Xte); probs = pipe.predict_proba(Xte)[:, 1]
    scores = [accuracy_score(yte, pred), precision_score(yte, pred, zero_division=0), recall_score(yte, pred, zero_division=0), f1_score(yte, pred, zero_division=0), roc_auc_score(yte, probs)]
    cols = st.columns(5)
    for col, label, val in zip(cols, ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"], scores):
        col.metric(label, f"{val:.3f}")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=["No purchase", "Purchase"], ax=ax)
    st.pyplot(fig); plt.close(fig)
    cid = st.selectbox("Choose a customer from the historical observation period", options=hx.index.astype(int).tolist())
    row = hx.loc[[cid], features]
    st.metric("Predicted outcome", "Purchase" if pipe.predict(row)[0] else "No purchase")
    st.metric("Estimated purchase probability", f"{pipe.predict_proba(row)[0, 1]:.1%}")
    st.caption("Educational retrospective demo. The displayed metrics use a stratified random split; a strictly later time-based holdout is recommended before making prospective claims.")

else:
    st.header("Model Comparison")
    st.dataframe(model_results, use_container_width=True)
    if "Model" in model_results.columns and "Accuracy" in model_results.columns:
        st.subheader("Accuracy by classifier")
        chart = model_results.set_index("Model")[["Accuracy"]] * 100
        st.bar_chart(chart)
    if "Model" in model_results.columns and "F1 Score" in model_results.columns:
        st.subheader("F1-score by classifier")
        st.bar_chart(model_results.set_index("Model")[["F1 Score"]])
    st.info("These are the model metrics exported from the Colab baseline experiment.")
