import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os

try:
    csv_train_path = os.path.join(os.path.dirname(__file__), "Germany 10-Year Bond Training Data.csv")
    csv_test_path = os.path.join(os.path.dirname(__file__), "Germany 10-Year Bond Test Data.csv")
    train_df = pd.read_csv(csv_train_path)
    test_df = pd.read_csv("Germany 10-Year Bond Test Data.csv")
except FileNotFoundError:
    print("Make sure the files 'Germany 10-Year Bond Training Data.csv' and 'Germany 10-Year Bond Test Data.csv' are available.")
    print("Please run the scripts Training_data_processing.py and Test_data_processing.py first.")
    exit()

features = [
    "Price Lag 1", "Price Lag 2", "Price Return 1d",
    "Momentum 3d", "MA 5d", "MA 10d", "Volatility 5d",
    "Change %"
]
target = "Price"

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target] 

model = RandomForestRegressor(n_estimators=300, random_state=42, oob_score=True)
print("Train the model...")
model.fit(X_train, y_train)
print("The model has been trained.")

print("Making predictions on the test data...")
predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

results_df = pd.DataFrame({
    "Date": test_df["Date"],
    "Actual Price": y_test,
    "Predicted Price": predictions
})
results_df["Difference"] = results_df["Actual Price"] - results_df["Predicted Price"]
print(results_df)

stats_text = (f"R²: {r2:.4f}  |  MAE: {mae:.4f}  |  MSE: {mse:.4f}  |  OOB Score: {model.oob_score_:.4f}")

st.set_page_config(layout="wide")
fig, ax = plt.subplots(figsize=(12, 6), dpi= 5)
fig.patch.set_facecolor("#000000")


ax.set_ylim(2.4, 2.8)
ax.plot(results_df["Date"], results_df["Predicted Price"], color="#03FFA7", 
        linewidth=2.5, alpha=0.9, label="Predicted Price", marker="o")
ax.plot(results_df["Date"], results_df["Actual Price"], color="#FFF702", 
        linewidth=2.5, alpha=0.9, label="Actual Price", marker="o")
ax.tick_params(colors='white', which='both')
ax.set_facecolor("#000000")
ax.grid(color = "#4D4E4E", linestyle= "-", linewidth=0.5)
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('#777777')
ax.spines['bottom'].set_color('#777777')

sns.set_style("whitegrid")
plt.title("Bond Comparison Between Prediction and Real-Life", color="white", fontsize= 14)
plt.ylabel("Price in Euro")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
# plt.figtext(1, 0.5, stats_text, ha="center", fontsize=10,
#             bbox={"facecolor":"#FFFBFB", "color": "white", "alpha":0.5, "pad":5})
plt.subplots_adjust(bottom=0.2) 
# plt.show()  # Comment out for Streamlit

# Set black to gray gradient theme for Streamlit
st.markdown("""
<style>
    .title-test {
        font-weight: bold;
        padding: 5px;
        border       
    }
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    .stApp > div {
        background-color: #000000;
    }
    .stMarkdown, .stText, h1, h2, h3, p {
        color: #FFFFFF !important;
    }
    .stTextInput > div > div > input {
        caret-color: white;
        background-color: #2D2D2D;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# Display the matplotlib figure in Streamlit with larger size
st.title("Germany 10-Year Bond Yield and Bond Price Prediction Using Random Forest Regression")
st.markdown("By Ignatius Arviant Darrell")
st.pyplot(fig, use_container_width=True)
st.markdown("---")
st.markdown("""
            
Here are the Model Performance Summary of the model:
- **R² Score**: 0.8766
The model explains almost 88% of the variation in bond prices, showing strong overall predictive power.

- **Mean Absolute Error (MAE)**: 0.0163
On average, the model's predictions differ by only 1.6 euro cents from the actual prices — indicating high accuracy.

- **Mean Squared Error (MSE)**: 0.0005
The very small error shows that large mistakes are rare, which is essential for financial forecasting.

- **Out-of-Bag (OOB) Score**: 0.9204
The model maintains high accuracy even on unseen data, proving it generalizes well without overfitting.

**Market Context**            

According to **Tradingview**, throughout June 2025, the German 10-year bond yield remained steady near 2.5% 
As investors closely monitored the economic outlook and future monetary policy direction, following a prolonged 
ceasefire between Iran and Israel. At the same time, NATO's plan to increase defence spending from 2% to 5% of GDP 
by 2035 has raised expectations of additional government borrowing — especially by Germany — to fulfil these new obligations.
            
For July, there was not much movement in monetary policy. According to Reuters, ECB President Christine Lagarde stated that she would only support a rate cut 
if there were “signs of a material deviation of inflation” from the 2% target. She also emphasised that the central bank should avoid “fine-tuning” 
interest rates in response to short-term fluctuations such as oil price swings.
            
This cautious tone was echoed by other policymakers. Additionally, The ECB's chief economist Philip Lane also said recently that the central bank would react 
to "material" changes in the euro zone's inflation outlook and ignore "tiny" ones.
            
**References:**
- https://www.tradingview.com/news/te_news:465790:0-germany-10-year-bond-yield-remains-near-2-5/
- https://www.reuters.com/business/finance/ecbs-schnabel-sets-bar-very-high-rate-cut-economy-holds-up-2025-07-11/

""")
