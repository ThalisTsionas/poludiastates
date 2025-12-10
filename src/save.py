from main import load_dataset, preprocess_dataset, build_base_pool

def main():
    #fortosi + preprocess (opos tin main)
    df_raw = load_dataset()
    df_proc = preprocess_dataset(df_raw)
    base_df = build_base_pool(df_proc)

    #apothikeuoume oles tis xores
    all_countries = set()

    if "origin_country_parsed" not in base_df.columns:
        raise RuntimeError("Η στήλη 'origin_country_parsed' δεν υπάρχει. Βεβαιώσου ότι τρέχεις το σωστό main.preprocess_dataset().")

    for lst in base_df["origin_country_parsed"]:
        if isinstance(lst, list):
            for c in lst:
                if isinstance(c, str) and c.strip():
                    all_countries.add(c.strip().upper())

    countries_sorted = sorted(all_countries)

    print("Total unique countries:", len(countries_sorted))
    print("Comma-separated:")
    print(", ".join(countries_sorted))
    print("\nPython list (για copy–paste στον κώδικα):")
    print(countries_sorted)

if __name__ == "__main__":
    main()

