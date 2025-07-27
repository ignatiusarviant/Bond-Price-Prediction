import pandas as pd

file = pd.read_csv("C:\\Users\\HP\\Downloads\\Python Only\\Bond Price ML\\Germany 10-Year Bond Yield Historical Data.csv")
df = pd.DataFrame(file)
df.drop(columns=["Open", "High", "Low"], inplace=True)

def interest_rate_addition(row):
    date = pd.to_datetime(row["Date"])
    if date <= pd.to_datetime("01/31/2025"):
        return round(248/100, 2)
    elif date <= pd.to_datetime("02/28/2025"):
        return round(240/100, 2)
    elif date <= pd.to_datetime("03/31/2025"):
        return round(274/100, 2)
    elif date <= pd.to_datetime("04/30/2025"):
        return round(251/100, 2)
    elif date <= pd.to_datetime("05/31/2025"):
        return round(256/100, 2)
    elif date <= pd.to_datetime("06/14/2025"):
        return round(252/100, 2)
    
def classification(x):
    if x < 0:
        return -1
    elif x == 0:
        return 0
    elif x > 0:
        return 1

df["Price Return 1d"] = df["Price"].astype(float).diff()
df["Price Lag 1"] = df["Price"].shift(1)
df["Price Lag 2"] = df["Price"].shift(2)
df["Momentum 3d"] = df["Price"].diff(3)
df["MA 5d"] = df["Price"].rolling(window=5).mean()
df["MA 10d"] = df["Price"].rolling(window=10).mean()
df["Volatility 5d"] = df["Price Return 1d"].rolling(window=5).std()
df["Interest Rate"] = df.apply(interest_rate_addition, axis=1)
df["Change %"] = df["Change %"].str.replace("%", "").astype(float)

df = df[[
    "Date", "Price", "Price Lag 1", "Price Lag 2", "Price Return 1d",
    "Momentum 3d", "MA 5d", "MA 10d", "Volatility 5d",
    "Change %", "Interest Rate"
]]

df["Date"] = pd.to_datetime(df["Date"])
filter = df["Date"] < pd.to_datetime("6/16/2025")
new_df = df[~filter].reset_index(drop=True)

print(new_df["Date"])
# new_df.to_csv("Germany 10-Year Bond Test Data.csv", index=False)