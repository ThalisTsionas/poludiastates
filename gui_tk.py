from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

from main import (
    load_dataset,
    preprocess_dataset,
    build_base_pool,
    apply_full_query_filters,
    AssignmentQueryParams,
    build_kd_index,
    build_quad_index,
    build_range_index,
    build_rtree_index,
    numeric_candidates_kdtree,
    numeric_candidates_quadtree,
    numeric_candidates_rangetree,
    numeric_candidates_rtree,
    extract_all_genres,
    build_genre_lsh_index,
    query_by_genre_name,
    evaluate_indexes,
)


class MovieQueryApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Movies - Trees + LSH")

        self.base_df: pd.DataFrame | None = None
        self.last_ground_truth_df: pd.DataFrame | None = None
        self.all_genres: list[str] = []

        self.status_var = tk.StringVar(value="Ready.")
        self.lsh_common_var = tk.StringVar(value="")

        self._build_layout()

    # ------------------------------------------------------------------
    # Layout (απλό)
    # ------------------------------------------------------------------
    def _build_layout(self):
        # --------- Filters frame ----------
        filters = ttk.LabelFrame(self.root, text="Filters")
        filters.pack(fill="x", padx=10, pady=5)

        # Year
        self.combo_year_min = self._labeled_combo(
            filters, "Year min:", 2000, row=0,
            values=[str(y) for y in range(1900, 2026)]
        )
        self.combo_year_max = self._labeled_combo(
            filters, "Year max:", 2020, row=0, col=2,
            values=[str(y) for y in range(1900, 2026)]
        )

        # Popularity
        pop_vals = [f"{x/2:.1f}" for x in range(0, 41)]
        self.combo_pop_min = self._labeled_combo(
            filters, "Popularity min:", "8.0", row=1, values=pop_vals
        )
        self.combo_pop_max = self._labeled_combo(
            filters, "Popularity max:", "12.0", row=1, col=2, values=pop_vals
        )

        # Vote avg
        vote_vals = [f"{x/2:.1f}" for x in range(0, 21)]
        self.combo_vote_min = self._labeled_combo(
            filters, "Vote min:", "3.0", row=2, values=vote_vals
        )
        self.combo_vote_max = self._labeled_combo(
            filters, "Vote max:", "8.0", row=2, col=2, values=vote_vals
        )

        # Runtime
        run_vals = [str(x) for x in range(30, 241, 10)]
        self.combo_runtime_min = self._labeled_combo(
            filters, "Runtime min:", "30", row=3, values=run_vals
        )
        self.combo_runtime_max = self._labeled_combo(
            filters, "Runtime max:", "180", row=3, col=2, values=run_vals
        )

        # Countries
        ttk.Label(filters, text="Countries:").grid(row=4, column=0, sticky="w")
        self.entry_countries = ttk.Entry(filters, width=18)
        self.entry_countries.insert(0, "US,GB")
        self.entry_countries.grid(row=4, column=1, padx=5, pady=2)

        # Language
        ttk.Label(filters, text="Language:").grid(row=4, column=2, sticky="w")
        self.combo_language = ttk.Combobox(filters, width=10, state="readonly",
                                           values=["en", "fr", "de", "es", "it", "ja"])
        self.combo_language.set("en")
        self.combo_language.grid(row=4, column=3, padx=5, pady=2)

        # Buttons
        ttk.Button(filters, text="Run query (pandas only)",
                   command=self.run_query).grid(row=5, column=0, columnspan=2, pady=8)
        ttk.Button(filters, text="Run index + LSH performance",
                   command=self.run_index_performance).grid(row=5, column=2, columnspan=2, pady=8)

        # --------- Tree-based LSH search frame ----------
        tree_lsh = ttk.LabelFrame(self.root, text="Tree-based LSH search (results shown in table)")
        tree_lsh.pack(fill="x", padx=10, pady=5)

        ttk.Label(tree_lsh, text="Index:").grid(row=0, column=0, sticky="w")
        self.combo_tree = ttk.Combobox(tree_lsh, width=15, state="readonly",
                                       values=["KD-Tree", "Quad-Tree", "Range-Tree", "R-Tree"])
        self.combo_tree.set("KD-Tree")
        self.combo_tree.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(tree_lsh, text="Genre:").grid(row=1, column=0, sticky="w")
        self.combo_genre = ttk.Combobox(tree_lsh, width=25, state="readonly", values=[])
        self.combo_genre.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(tree_lsh, text="Top-N:").grid(row=1, column=2, sticky="w")
        self.entry_topn = ttk.Entry(tree_lsh, width=5)
        self.entry_topn.insert(0, "5")
        self.entry_topn.grid(row=1, column=3, padx=5, pady=2)

        ttk.Button(tree_lsh, text="Run Tree + LSH search",
                   command=self.run_tree_lsh).grid(row=2, column=0, columnspan=4, pady=6)

        # --------- Status ----------
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", padx=10, pady=3)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side="left")
        ttk.Label(status_frame, textvariable=self.lsh_common_var).pack(side="right")

        # --------- Results table ----------
        results = ttk.LabelFrame(self.root, text="Movies")
        results.pack(fill="both", expand=True, padx=10, pady=5)

        cols = ["title", "year", "popularity", "vote"]
        self.tree_results = ttk.Treeview(results, columns=cols, show="headings", height=10)
        for c in cols:
            self.tree_results.heading(c, text=c)
            width = 300 if c == "title" else 80
            self.tree_results.column(c, width=width, anchor="w")
        self.tree_results.pack(fill="both", expand=True)

        # --------- Numeric performance table ----------
        perf_num = ttk.LabelFrame(self.root, text="Numeric index performance")
        perf_num.pack(fill="x", padx=10, pady=5)

        num_cols = ["Index", "Build_s", "Numeric_s", "Candidates"]
        self.tree_perf_numeric = ttk.Treeview(perf_num, columns=num_cols, show="headings", height=5)
        for c in num_cols:
            self.tree_perf_numeric.heading(c, text=c)
            self.tree_perf_numeric.column(c, width=100, anchor="center")
        self.tree_perf_numeric.pack(fill="x", expand=False)

        # --------- Scheme performance table ----------
        perf_scheme = ttk.LabelFrame(self.root, text="Schemes: (Index + LSH)")
        perf_scheme.pack(fill="x", padx=10, pady=5)

        scheme_cols = ["Scheme", "IdxBuild_s", "NumQ_s", "Cand", "Res", "LSH_build_s", "LSH_query_s"]
        self.tree_perf_scheme = ttk.Treeview(perf_scheme, columns=scheme_cols, show="headings", height=6)
        for c in scheme_cols:
            self.tree_perf_scheme.heading(c, text=c)
            self.tree_perf_scheme.column(c, width=110, anchor="center")
        self.tree_perf_scheme.pack(fill="x", expand=False)

    # ------------------------------------------------------------------
    # Helper για label + combobox
    # ------------------------------------------------------------------
    def _labeled_combo(self, parent, label, default, row, col=0, values=()):
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=2, pady=2)
        cb = ttk.Combobox(parent, width=10, state="readonly", values=values)
        cb.set(str(default))
        cb.grid(row=row, column=col + 1, padx=2, pady=2)
        return cb

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def ensure_base_df_loaded(self):
        if self.base_df is not None:
            return
        try:
            self.status_var.set("Loading dataset...")
            self.root.update_idletasks()
            df_raw = load_dataset()
            df_proc = preprocess_dataset(df_raw)
            self.base_df = build_base_pool(df_proc)
            self.status_var.set(f"Base pool loaded: {len(self.base_df)} movies.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
            self.status_var.set("Error loading data.")
            self.base_df = None

    # ------------------------------------------------------------------
    # Read & validate parameters
    # ------------------------------------------------------------------
    def get_params_from_ui(self) -> AssignmentQueryParams | None:
        try:
            year_min = int(self.combo_year_min.get())
            year_max = int(self.combo_year_max.get())
            pop_min = float(self.combo_pop_min.get())
            pop_max = float(self.combo_pop_max.get())
            vote_min = float(self.combo_vote_min.get())
            vote_max = float(self.combo_vote_max.get())
            runtime_min = int(self.combo_runtime_min.get())
            runtime_max = int(self.combo_runtime_max.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid numeric values in filters.")
            return None

        # Validation min <= max
        if year_max < year_min:
            messagebox.showerror("Error", "Year max must be >= Year min.")
            return None
        if pop_max < pop_min:
            messagebox.showerror("Error", "Popularity max must be >= Popularity min.")
            return None
        if vote_max < vote_min:
            messagebox.showerror("Error", "Vote max must be >= Vote min.")
            return None
        if runtime_max < runtime_min:
            messagebox.showerror("Error", "Runtime max must be >= Runtime min.")
            return None

        countries_text = self.entry_countries.get().strip()
        if countries_text:
            countries = tuple(c.strip().upper() for c in countries_text.split(",") if c.strip())
        else:
            countries = tuple()

        language = self.combo_language.get().strip().lower() or "en"

        return AssignmentQueryParams(
            year_min=year_min,
            year_max=year_max,
            pop_min=pop_min,
            pop_max=pop_max,
            vote_min=vote_min,
            vote_max=vote_max,
            runtime_min=runtime_min,
            runtime_max=runtime_max,
            allowed_countries=countries,
            language=language,
        )

    # ------------------------------------------------------------------
    # Query (μόνο pandas, για να βλέπουμε αποτελέσματα και genres)
    # ------------------------------------------------------------------
    def run_query(self):
        self.ensure_base_df_loaded()
        if self.base_df is None:
            return

        params = self.get_params_from_ui()
        if params is None:
            return

        self.status_var.set("Running query (pandas)...")
        self.root.update_idletasks()

        df = apply_full_query_filters(self.base_df, params)
        self.last_ground_truth_df = df

        self.update_movies_table(df)
        self.status_var.set(f"Query done. Found {len(df)} movies.")

        # Update genres for genre combobox (used by tree-based LSH)
        self.all_genres = extract_all_genres(df)
        self.combo_genre["values"] = self.all_genres
        if self.all_genres:
            self.combo_genre.set(self.all_genres[0])
        else:
            self.combo_genre.set("")

    # ------------------------------------------------------------------
    # Tree-based LSH search (ο χρήστης βλέπει top-N αποτελέσματα)
    # ------------------------------------------------------------------
    def run_tree_lsh(self):
        self.ensure_base_df_loaded()
        if self.base_df is None:
            return

        params = self.get_params_from_ui()
        if params is None:
            return

        index_type = self.combo_tree.get().strip()
        genre = self.combo_genre.get().strip()
        if not genre:
            messagebox.showwarning("Warning", "Please select a genre.")
            return

        try:
            top_n = int(self.entry_topn.get())
        except ValueError:
            messagebox.showerror("Error", "Top-N must be an integer.")
            return

        self.status_var.set(f"Running Tree + LSH search using {index_type} ...")
        self.root.update_idletasks()

        # 1. Numeric filtering ανά index
        if index_type == "KD-Tree":
            features = ["release_year", "popularity", "vote_average", "runtime", "vote_count"]
            kd_tree, _ = build_kd_index(self.base_df, features)
            numeric_df, _ = numeric_candidates_kdtree(kd_tree, self.base_df, params, features)

        elif index_type == "Quad-Tree":
            quad_tree, _ = build_quad_index(self.base_df)
            numeric_df, _ = numeric_candidates_quadtree(quad_tree, self.base_df, params)

        elif index_type == "Range-Tree":
            year_tree, _ = build_range_index(self.base_df)
            numeric_df, _ = numeric_candidates_rangetree(year_tree, self.base_df, params)

        elif index_type == "R-Tree":
            rtree_obj, _ = build_rtree_index(self.base_df)
            numeric_df, _ = numeric_candidates_rtree(rtree_obj, self.base_df, params)
        else:
            messagebox.showerror("Error", "Unknown index type.")
            return

        # 2. Category filtering
        final_df = apply_full_query_filters(numeric_df, params)

        if final_df.empty:
            self.status_var.set("No movies after numeric + categorical filters.")
            self.update_movies_table(final_df)
            return

        # 3. Build LSH πάνω στο subset
        genre_index = build_genre_lsh_index(final_df)

        # 4. LSH query
        result_df = query_by_genre_name(genre_index, genre, top_n)
        self.update_movies_table(result_df)

        self.status_var.set(
            f"Tree + LSH search done ({index_type}), Genre='{genre}', returned {len(result_df)} movies."
        )

    # ------------------------------------------------------------------
    # Index + LSH performance (2 πίνακες)
    # ------------------------------------------------------------------
    def run_index_performance(self):
        self.ensure_base_df_loaded()
        if self.base_df is None:
            return

        params = self.get_params_from_ui()
        if params is None:
            return

        self.status_var.set("Running index + LSH performance...")
        self.root.update_idletasks()

        # ground truth = πλήρες αποτέλεσμα με pandas
        ground_truth_df = apply_full_query_filters(self.base_df, params)
        self.last_ground_truth_df = ground_truth_df

        # Χρησιμοποιούμε την evaluate_indexes από το main.py
        summary_numeric, summary_schemes, lsh_common_build, lsh_common_query = evaluate_indexes(
            self.base_df, ground_truth_df, params
        )

        # Ενημέρωση πινάκων
        self.update_numeric_perf_table(summary_numeric)
        self.update_scheme_perf_table(summary_schemes)

        self.lsh_common_var.set(
            f"Common LSH on GT (|GT|={len(ground_truth_df)}): "
            f"build={lsh_common_build:.4f}s, query={lsh_common_query:.6f}s"
        )

        self.status_var.set("Index + LSH performance done.")

    # ------------------------------------------------------------------
    # Table helpers
    # ------------------------------------------------------------------
    def update_movies_table(self, df: pd.DataFrame):
        for row in self.tree_results.get_children():
            self.tree_results.delete(row)

        if df is None or df.empty:
            return

        view_df = df.head(200)
        for _, r in view_df.iterrows():
            self.tree_results.insert(
                "",
                "end",
                values=[
                    str(r.get("title", "")),
                    str(r.get("release_year", "")),
                    f"{float(r.get('popularity', 0.0)):.2f}",
                    f"{float(r.get('vote_average', 0.0)):.1f}",
                ],
            )

    def update_numeric_perf_table(self, summary_numeric: dict):
        for row in self.tree_perf_numeric.get_children():
            self.tree_perf_numeric.delete(row)

        for name, stats in summary_numeric.items():
            self.tree_perf_numeric.insert(
                "",
                "end",
                values=[
                    name,
                    f"{stats['build']:.4f}",
                    f"{stats['numeric']:.4f}",
                    str(stats['numeric_candidates']),
                ],
            )

    def update_scheme_perf_table(self, summary_schemes: dict):
        for row in self.tree_perf_scheme.get_children():
            self.tree_perf_scheme.delete(row)

        for name, stats in summary_schemes.items():
            self.tree_perf_scheme.insert(
                "",
                "end",
                values=[
                    name,
                    f"{stats['build_index']:.4f}",
                    f"{stats['numeric_time']:.4f}",
                    str(stats['numeric_candidates']),
                    str(stats['result_size']),
                    f"{stats['lsh_build']:.4f}",
                    f"{stats['lsh_query']:.6f}",
                ],
            )


def main():
    root = tk.Tk()
    app = MovieQueryApp(root)
    root.geometry("1000x900")
    root.mainloop()


if __name__ == "__main__":
    main()
