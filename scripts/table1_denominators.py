"""Print exact n/N for the Table 1 binary rows, overall and per sex."""
from oral_mb.data import load_cohort

ROWS = (
    "abscess_or_gingivitis",
    "bleeding_gums",
    "bad_breath_lose",
    "bad_breath_strict",
)


def fmt(n, total):
    return f"{n}/{total} ({100 * n / total:.1f}%)"


def main():
    c = load_cohort()
    w = c.covariates[["gender"]].join(c.symptoms, how="left")
    counts = w["gender"].value_counts(dropna=False).to_dict()
    print(f"header N: overall {len(w)}")
    print(f"header N by gender code: {counts}")
    for col in ROWS:
        s = w[[col, "gender"]].dropna(subset=[col])
        print(f"\n{col}")
        print(f"   overall: {fmt(int(s[col].sum()), len(s))}")
        by = s.groupby("gender", dropna=False)[col].agg(["sum", "count"])
        for g, r in by.iterrows():
            print(f"   gender={g}: {fmt(int(r['sum']), int(r['count']))}")


if __name__ == "__main__":
    main()
