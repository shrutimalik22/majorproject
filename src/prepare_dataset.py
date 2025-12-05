import pandas as pd
from pathlib import Path
import os

def main():
    # Debug: show where we are
    print("[DEBUG] CWD:", os.getcwd())

    data_dir = Path("data")
    print("[DEBUG] Files in data/:", [p.name for p in data_dir.glob("*.csv")])

    monday_path = data_dir / "monday.csv"
    fri_morning_path = data_dir / "friday_morning.csv"
    fri_ddos_path = data_dir / "friday_ddos.csv"
    fri_portscan_path = data_dir / "friday_portscan.csv"

    print("[INFO] Loading CSV files...")

    df_monday = pd.read_csv(monday_path)
    print("[INFO] Monday (Normal) shape:", df_monday.shape)

    df_fmorning = pd.read_csv(fri_morning_path)
    print("[INFO] Friday Morning (Attacks) shape:", df_fmorning.shape)

    df_fddos = pd.read_csv(fri_ddos_path)
    print("[INFO] Friday DDoS (Attacks) shape:", df_fddos.shape)

    df_fport = pd.read_csv(fri_portscan_path)
    print("[INFO] Friday PortScan (Attacks) shape:", df_fport.shape)

    # Add simple binary labels
    print("[INFO] Assigning binary labels...")
    df_monday["Label"] = "Normal"
    df_fmorning["Label"] = "Attack"
    df_fddos["Label"] = "Attack"
    df_fport["Label"] = "Attack"

    # Keep only common columns across all 4
    common_cols = set(df_monday.columns)
    common_cols &= set(df_fmorning.columns)
    common_cols &= set(df_fddos.columns)
    common_cols &= set(df_fport.columns)

    common_cols = sorted(list(common_cols))
    print("[INFO] Number of common columns:", len(common_cols))

    df_monday = df_monday[common_cols]
    df_fmorning = df_fmorning[common_cols]
    df_fddos = df_fddos[common_cols]
    df_fport = df_fport[common_cols]

    print("[INFO] Concatenating all data...")
    df_all = pd.concat(
        [df_monday, df_fmorning, df_fddos, df_fport],
        ignore_index=True
    )

    print("[INFO] Final combined shape:", df_all.shape)
    print("[INFO] Label distribution:")
    print(df_all["Label"].value_counts())

    output_path = data_dir / "cicids_phase1.csv"
    df_all.to_csv(output_path, index=False)
    print("[INFO] Saved combined dataset to:", output_path)

if __name__ == "__main__":
    main()
