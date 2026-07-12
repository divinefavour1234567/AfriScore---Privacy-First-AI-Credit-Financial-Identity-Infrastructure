"""
AfriScore - Synthetic Alternative Data Generator
Creates realistic alt-data profiles for Nigerian/African users (youth, MSMEs, smallholders).
No real personal data — fully synthetic for hackathon demo.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import pathlib

np.random.seed(42)
random.seed(42)

def generate_synthetic_users(n_users=500):
    data = []
    
    for i in range(n_users):
        user_id = f"AFRI_{str(i).zfill(6)}"
        age = np.random.randint(18, 45)
        
        # Location bias (more urban for demo diversity)
        location_type = np.random.choice(["Urban", "Semi-Urban", "Rural"], p=[0.45, 0.35, 0.2])
        state = random.choice(["Lagos", "Abuja", "Enugu", "Kano", "Port Harcourt", "Ibadan", "Other"])
        
        # Mobile Money / Digital Footprint (strong signal)
        mobile_txn_count_6m = np.random.randint(20, 450) if age < 35 else np.random.randint(10, 300)
        avg_txn_amount = np.random.uniform(1500, 45000)  # Naira
        txn_regularity = np.clip(np.random.normal(0.75, 0.18), 0.3, 1.0)  # consistency score
        
        # Utility & Bill Payments (important for thin-file users)
        utility_consistency = np.clip(np.random.normal(0.68, 0.22), 0.2, 1.0)
        has_electricity_payment = np.random.choice([0, 1], p=[0.25, 0.75])
        
        # Education & Skills (ties to Future of Work angle)
        education_level = np.random.choice([0, 1, 2, 3], p=[0.15, 0.35, 0.35, 0.15])  # 0=None, 1=Secondary, 2=OND/HND, 3=Degree+
        skills_completed = np.random.randint(0, 12) if education_level >= 1 else np.random.randint(0, 4)
        
        # Behavioral / Savings Proxy (from transaction patterns)
        savings_rate_proxy = np.clip(np.random.normal(0.22, 0.12), 0.02, 0.55)
        app_engagement_score = np.clip(np.random.normal(0.71, 0.19), 0.25, 1.0)
        
        # Smallholder / Agri signals (for future expansion)
        is_smallholder = 1 if location_type == "Rural" and np.random.rand() > 0.6 else 0
        farm_size_proxy = np.random.uniform(0.5, 8.0) if is_smallholder else 0
        yield_risk_score = np.clip(np.random.normal(0.45, 0.25), 0.1, 0.9) if is_smallholder else 0.5
        
        # Target variable for training (simulated "creditworthy")
        # Higher score = more likely creditworthy based on positive signals
        base_score = (
            (mobile_txn_count_6m / 400) * 0.25 +
            (txn_regularity * 0.20) +
            (utility_consistency * 0.18) +
            (education_level / 3 * 0.12) +
            (skills_completed / 12 * 0.10) +
            (savings_rate_proxy * 0.10) +
            (app_engagement_score * 0.05)
        )
        
        # Add noise and smallholder adjustment
        noise = np.random.normal(0, 0.08)
        creditworthy_prob = np.clip(base_score + noise - (yield_risk_score * 0.1 if is_smallholder else 0), 0, 1)
        is_creditworthy = 1 if creditworthy_prob > 0.55 else 0
        
        # Derived "true" risk score for demo (0-1000 scale)
        risk_score = int(np.clip( (base_score * 850) + np.random.normal(50, 80), 280, 920 ))
        
        data.append({
            "user_id": user_id,
            "age": age,
            "location_type": location_type,
            "state": state,
            "mobile_txn_count_6m": mobile_txn_count_6m,
            "avg_txn_amount": round(avg_txn_amount, 0),
            "txn_regularity_score": round(txn_regularity, 3),
            "utility_consistency": round(utility_consistency, 3),
            "has_electricity_payment": has_electricity_payment,
            "education_level": education_level,
            "skills_completed": skills_completed,
            "savings_rate_proxy": round(savings_rate_proxy, 3),
            "app_engagement_score": round(app_engagement_score, 3),
            "is_smallholder": is_smallholder,
            "farm_size_proxy": round(farm_size_proxy, 1),
            "yield_risk_score": round(yield_risk_score, 3),
            "creditworthy_label": is_creditworthy,
            "true_risk_score": risk_score
        })
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = generate_synthetic_users(600)
    # Save relative to project root (parent of backend/)
    script_dir = pathlib.Path(__file__).parent.parent.resolve()
    output_path = script_dir / "data" / "synthetic_users.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} synthetic users.")
    print(f"Saved to: {output_path}")
    print("\nSample columns:", list(df.columns))
    print("\nFirst 3 rows preview:")
    print(df.head(3).to_string())