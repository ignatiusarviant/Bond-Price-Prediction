import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

try:
    train_df = pd.read_csv("C:\\Users\\HP\\Downloads\\Python Only\\Bond Price ML\\Germany 10-Year Bond Training Data.csv")
    test_df = pd.read_csv("C:\\Users\\HP\\Downloads\\Python Only\\Bond Price ML\\Germany 10-Year Bond Test Data.csv")
except FileNotFoundError:
    print("Make sure the files 'Germany 10-Year Bond Training Data.csv' and 'Germany 10-Year Bond Test Data.csv' are available.")
    print("Please run the scripts Training_data_processing.py and Test_data_processing.py first.")
    exit()

features = [
    "Price Lag 1", "Price Lag 2", "Price Return 1d",
    "Momentum 3d", "MA 5d", "MA 10d", "Volatility 5d",
    "Change %", "Interest Rate"
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
fig, ax = plt.subplots(figsize=(14, 7))
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
plt.figtext(0.5, 0.05, stats_text, ha="center", fontsize=10,
            bbox={"facecolor":"#FFFBFB", "color": "white", "alpha":0.5, "pad":5})
plt.subplots_adjust(bottom=0.2) 
plt.show()