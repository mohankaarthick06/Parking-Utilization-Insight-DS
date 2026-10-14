import os
import random
import datetime
import numpy as np
import pandas as pd

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

print("--- Generating Calibrated Raw Datasets ---")

# Zones definition
zones_data = [
    {"zone": "Zone_A", "location": "Downtown Financial Center", "capacity": 150, "zone_type": "Commercial"},
    {"zone": "Zone_B", "location": "Metro Shopping Plaza", "capacity": 200, "zone_type": "Commercial"},
    {"zone": "Zone_C", "location": "City General Hospital", "capacity": 120, "zone_type": "Special Purpose"},
    {"zone": "Zone_D", "location": "Riverside Arts & Arena", "capacity": 250, "zone_type": "Event/Leisure"},
    {"zone": "Zone_E", "location": "Northside Residential Enclave", "capacity": 180, "zone_type": "Residential"},
    {"zone": "Zone_F", "location": "Tech Park Innovation Hub", "capacity": 220, "zone_type": "Commercial"},
    {"zone": "Zone_G", "location": "Suburban Transit Terminal", "capacity": 300, "zone_type": "Transit / Park & Ride"},
    {"zone": "Zone_H", "location": "Old Town Cultural Market", "capacity": 90, "zone_type": "Commercial"},
    # Intentional capacity mismatch record in raw source per rubric
    {"zone": "Zone_A", "location": "Downtown Financial Center", "capacity": 50, "zone_type": "Commercial"},
]
df_zones = pd.DataFrame(zones_data)
df_zones.to_csv("data/raw/zones.csv", index=False)

# Date range: 30 days
start_date = datetime.date(2026, 9, 1)
date_list = [start_date + datetime.timedelta(days=i) for i in range(30)]

# Calendar data
calendar_rows = []
for d in date_list:
    weekday = d.strftime("%A")
    is_weekend = 1 if weekday in ["Saturday", "Sunday"] else 0
    is_holiday = 1 if d.day in [7, 21] else 0
    # Major arena event dates
    is_event = 1 if d.day in [5, 12, 19, 26, 27] else 0
    event_name = "Stadium Concert" if d.day in [5, 19] else ("League Football Match" if d.day in [12, 26, 27] else "None")
    calendar_rows.append({
        "date": d.strftime("%Y-%m-%d"),
        "weekday": weekday,
        "is_weekend": is_weekend,
        "holiday": is_holiday,
        "event": event_name
    })
df_calendar = pd.DataFrame(calendar_rows)
df_calendar.to_csv("data/raw/calendar.csv", index=False)

# Weather data (hourly for the 30 days)
weather_rows = []
for d in date_list:
    date_str = d.strftime("%Y-%m-%d")
    is_rainy_day = 1 if d.day in [3, 8, 9, 15, 22, 23] else 0
    for hour in range(24):
        base_temp = 20 + 7 * np.sin((hour - 8) * np.pi / 12) + np.random.normal(0, 1.0)
        rain_val = 0.0
        if is_rainy_day and 10 <= hour <= 20:
            rain_val = round(float(np.random.gamma(shape=3.0, scale=3.5)), 1)
        weather_rows.append({
            "date": date_str,
            "hour": hour,
            "rain_mm": rain_val,
            "temperature_c": round(float(base_temp), 1)
        })
df_weather = pd.DataFrame(weather_rows)
df_weather.to_csv("data/raw/weather.csv", index=False)

# Dirty label mapping for real-world messy data per rubric
dirty_label_map = {
    "Zone_A": ["Zone_A", "zone_a", "ZONE A", "Zone A "],
    "Zone_B": ["Zone_B", "zone_b", "ZONE_B", "Zone B"],
    "Zone_C": ["Zone_C", "Zone_c", "zone c"],
    "Zone_D": ["Zone_D", "zone_d", "ZONE D "],
    "Zone_E": ["Zone_E", "zone_e", "Zone E"],
    "Zone_F": ["Zone_F", "zone_f"],
    "Zone_G": ["Zone_G", "zone_g", "Zone G "],
    "Zone_H": ["Zone_H", "zone_h"]
}

sessions = []
session_id_counter = 20001

for d in date_list:
    date_str = d.strftime("%Y-%m-%d")
    day_cal = df_calendar[df_calendar["date"] == date_str].iloc[0]
    weekday = day_cal["weekday"]
    is_weekend = day_cal["is_weekend"]
    has_event = day_cal["event"] != "None"
    is_rain = d.day in [3, 8, 9, 15, 22, 23]
    
    # Generate realistic hourly arrivals for each zone
    for z_info in zones_data[:8]:
        z = z_info["zone"]
        cap = z_info["capacity"]
        ztype = z_info["zone_type"]
        
        for hour in range(24):
            # Baseline arrival rate multiplier based on zone type and time
            if z in ["Zone_A", "Zone_F"]:  # Downtown / Tech Park (Business)
                if is_weekend:
                    lam = cap * 0.08 if 11 <= hour <= 20 else cap * 0.02
                else:
                    if 8 <= hour <= 10:    # Morning rush
                        lam = cap * 0.32
                    elif 11 <= hour <= 14: # Lunch rush
                        lam = cap * 0.28
                    elif 17 <= hour <= 19: # Evening commute
                        lam = cap * 0.18
                    elif 21 <= hour or hour <= 6:
                        lam = cap * 0.02
                    else:
                        lam = cap * 0.12
                dur_mean = 180 if not is_weekend else 120
                
            elif z == "Zone_B":  # Shopping Plaza
                if is_weekend:
                    lam = cap * 0.35 if 12 <= hour <= 20 else cap * 0.08
                else:
                    lam = cap * 0.22 if 12 <= hour <= 20 else cap * 0.06
                dur_mean = 135
                
            elif z == "Zone_D":  # Arena / Event
                if has_event and 17 <= hour <= 20:
                    lam = cap * 0.45  # Massive rush
                    dur_mean = 240
                elif has_event and 21 <= hour <= 23:
                    lam = cap * 0.05
                    dur_mean = 90
                else:
                    lam = cap * 0.05 if 10 <= hour <= 20 else cap * 0.01
                    dur_mean = 90
                    
            elif z == "Zone_E":  # Residential
                if 17 <= hour <= 22:
                    lam = cap * 0.18
                    dur_mean = 540  # Overnight stay (9 hours)
                elif 8 <= hour <= 16:
                    lam = cap * 0.04
                    dur_mean = 120
                else:
                    lam = cap * 0.02
                    dur_mean = 300
                    
            elif z == "Zone_G":  # Transit / Park & Ride
                if not is_weekend and 7 <= hour <= 9:
                    lam = cap * 0.25
                    dur_mean = 510  # Full work day stay (8.5 hrs)
                elif not is_weekend and 10 <= hour <= 16:
                    lam = cap * 0.05
                    dur_mean = 240
                else:
                    lam = cap * 0.02
                    dur_mean = 120
                    
            elif z == "Zone_C":  # Hospital
                lam = cap * 0.18 if 8 <= hour <= 18 else cap * 0.06
                dur_mean = 150
                
            else:  # Zone_H: Old Town Cultural Market
                lam = cap * 0.25 if (is_weekend and 10 <= hour <= 18) else cap * 0.12
                dur_mean = 110

            # Rainy day adjustment: reduces outdoor/on-street parking
            if is_rain and 10 <= hour <= 20 and z in ["Zone_A", "Zone_H"]:
                lam *= 0.74  # 26% drop

            # Sample arrivals
            num_arrivals = np.random.poisson(lam=max(0.5, lam))
            for _ in range(num_arrivals):
                minute = np.random.randint(0, 60)
                sec = np.random.randint(0, 60)
                entry_dt = datetime.datetime(d.year, d.month, d.day, hour, minute, sec)
                
                # Sample duration with variability
                duration_mins = max(15, int(np.random.exponential(dur_mean)))
                # Capped at 1440 mins
                duration_mins = min(duration_mins, 1440)
                exit_dt = entry_dt + datetime.timedelta(minutes=duration_mins)
                
                # Zone label casing noise
                zone_label = random.choice(dirty_label_map[z]) if random.random() < 0.20 else z
                
                # Fee calculation
                rate = 3.5 if z in ["Zone_A", "Zone_B"] else (5.0 if z == "Zone_D" and has_event else 2.0)
                fee = round(max(2.0, (duration_mins / 60.0) * rate), 2)
                
                sessions.append({
                    "session_id": f"SES_{session_id_counter}",
                    "zone": zone_label,
                    "entry": entry_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "exit": exit_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration": duration_mins,
                    "fee": fee
                })
                session_id_counter += 1

df_sessions = pd.DataFrame(sessions)

# Inject intentional Data Quality issues per rubric:
# 1. Missing exit times (~2.5%)
missing_exit_idx = np.random.choice(df_sessions.index, size=int(len(df_sessions) * 0.025), replace=False)
df_sessions.loc[missing_exit_idx, "exit"] = np.nan
df_sessions.loc[missing_exit_idx[:len(missing_exit_idx)//2], "duration"] = np.nan

# 2. Duplicate sessions (~2.0%)
dup_rows = df_sessions.sample(n=int(len(df_sessions) * 0.02), random_state=42)
df_sessions = pd.concat([df_sessions, dup_rows], ignore_index=True)

# 3. Negative durations (~0.6% errors)
neg_idx = np.random.choice(df_sessions.index, size=int(len(df_sessions) * 0.006), replace=False)
df_sessions.loc[neg_idx, "duration"] = -1 * np.random.randint(15, 240, size=len(neg_idx))

df_sessions.to_csv("data/raw/parking_sessions.csv", index=False)

# Payments data
payments = []
payment_modes = ["Credit Card", "Mobile App", "NFC / Tap", "Cash", "Fleet Account"]
weights = [0.48, 0.34, 0.11, 0.04, 0.03]

for _, row in df_sessions.iterrows():
    amt = row["fee"] if pd.notna(row["fee"]) and row["fee"] > 0 else 5.0
    payments.append({
        "payment_id": f"PAY_{row['session_id']}",
        "session_id": row["session_id"],
        "payment_mode": np.random.choice(payment_modes, p=weights),
        "amount": amt
    })
df_payments = pd.DataFrame(payments)
df_payments.to_csv("data/raw/payments.csv", index=False)

print(f"Complete! Generated {len(df_sessions)} sessions, {len(df_zones)} zone records, {len(df_calendar)} calendar days, {len(df_weather)} weather readings, {len(df_payments)} payments.")
