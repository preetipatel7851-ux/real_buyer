"""Creates a realistic sample dataset (data/raw/clients.csv).

Skip this if you already have your own file: just save it as
data/raw/clients.csv with the same 10 column names.
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N = 3000

REGIONS = {
    "United States": ["North America - East", "North America - West"],
    "Canada": ["North America - East"],
    "United Kingdom": ["Europe - West"],
    "Germany": ["Europe - West"],
    "France": ["Europe - West"],
    "United Arab Emirates": ["Middle East"],
    "India": ["South Asia"],
    "Singapore": ["Southeast Asia"],
    "Australia": ["Oceania"],
    "Brazil": ["Latin America"],
}
countries = rng.choice(list(REGIONS), N, p=[.22, .06, .12, .08, .07, .12, .13, .08, .07, .05])
regions = [rng.choice(REGIONS[c]) for c in countries]

client_type = rng.choice(["Individual", "Corporate"], N, p=[.78, .22])
gender = np.where(client_type == "Corporate", "Not Specified",
                  rng.choice(["Male", "Female", "Other"], N, p=[.52, .45, .03]))

age = np.where(client_type == "Corporate", rng.integers(30, 66, N),
               np.clip(rng.normal(41, 11, N), 22, 75)).astype(int)
dob = pd.Timestamp("2026-09-23") - pd.to_timedelta(age * 365 + rng.integers(0, 365, N), unit="D")

# Investment likelihood: corporate & mid-age & Middle East/Asia more likely to invest
p_invest = 0.35 + 0.4 * (client_type == "Corporate") + 0.1 * ((age > 35) & (age < 55))
p_invest += 0.08 * np.isin(countries, ["United Arab Emirates", "Singapore", "India"])
purpose = np.where(rng.random(N) < np.clip(p_invest, 0, .95), "Investment", "Personal use")

p_loan = np.where(purpose == "Personal use", 0.72, 0.38) - 0.25 * (client_type == "Corporate")
loan = np.where(rng.random(N) < np.clip(p_loan, .05, .95), "Yes", "No")

channel = rng.choice(["Website", "Agent Referral", "Social Media", "Friend/Family", "Property Portal", "Event"],
                     N, p=[.2, .25, .18, .14, .17, .06])

sat = np.clip(np.round(rng.normal(3.9, .8, N) + 0.3 * (channel == "Agent Referral")
                       - 0.3 * (loan == "Yes")), 1, 5).astype(int)

df = pd.DataFrame({
    "client_id": [f"CL{100000 + i}" for i in range(N)],
    "client_type": client_type, "gender": gender, "country": countries, "region": regions,
    "date_of_birth": dob.strftime("%Y-%m-%d"), "acquisition_purpose": purpose,
    "loan_applied": loan, "referral_channel": channel, "satisfaction_score": sat,
})
# a few realistic data-quality issues so the cleaning step is meaningful
for col in ["satisfaction_score", "referral_channel"]:
    df.loc[rng.choice(N, 40, replace=False), col] = np.nan
df = pd.concat([df, df.sample(15, random_state=1)])  # duplicates

Path("data/raw").mkdir(parents=True, exist_ok=True)
df.to_csv("data/raw/clients.csv", index=False)
print(f"Saved data/raw/clients.csv with {len(df)} rows")
