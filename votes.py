import pandas as pd
from pathlib import Path

COLS = ["rm", "beteckning", "votering_id", "punkt", "namn", "intressent_id",
        "parti", "valkrets", "rost", "avser", "banknummer", "kon",
        "fodd", "datum"]

def load_votes(folder):
    """Load riksdagen's vote CSVs and aggregate to one row per
    (rm, beteckning, punkt, parti, rost)."""
    frames = []
    for path in sorted(Path(folder).rglob("*.csv")):
        df = pd.read_csv(path, header=None, names=COLS, dtype=str)
        frames.append(df)
    votes = pd.concat(frames, ignore_index=True)
    votes = votes[votes["avser"] == "sakfrågan"]
    votes["punkt"] = pd.to_numeric(votes["punkt"], errors="coerce")
    return (votes
            .groupby(["rm", "beteckning", "punkt", "parti", "rost"])
            .size()
            .reset_index(name="antal"))

if __name__ == "__main__":
    import sys
    agg = load_votes(sys.argv[1])
    print(agg.shape)
    print(agg.head(12).to_string(index=False))
    print("\nunika voteringar:", agg[["rm", "beteckning", "punkt"]].drop_duplicates().shape[0])
    print("röstvärden:", sorted(agg["rost"].unique()))