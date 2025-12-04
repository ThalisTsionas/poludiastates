from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

# Import από το main.py (το clean που έχεις τώρα)
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
    run_lsh_on_df,
    # ΝΕΑ imports για genres + LSH
    extract_all_genres,
    build_genre_lsh_index,
    query_by_genre_name,
    GenreLSHIndex,
)


def treeview_sort_column(tv: ttk.Treeview, col: str, reverse: bool):
    """Sort στα headers του πίνακα."""
    data = [(tv.set(k, col), k) for k in tv.get_children("")]
    def try_float(s):
        try:
            return float(s)
        except ValueError:
            return s
    data = [(try_float(v), k) for v, k in data]
    data.sort(reverse=reverse)
    for index, (_, k) in enumerate(data):
        tv.move(k, "", index)

    tv.heading(col, command=lambda: treeview_sort_column(tv, col, not reverse))


class MovieQueryApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Movies Query & Index Performance")

        self.status_var = tk.StringVar(value="Ready.")
        self.lsh_common_var = tk.StringVar(value="Common LSH: -")

        self.base_df: pd.DataFrame | None = None
        self.last_ground_truth_df: pd.DataFrame | None = None

        # ΝΕΑ: για genre LSH
        self.all_genres: list[str] | None = None
        self.genre_index: GenreLSHIndex | None = None

        self._build_layout()

    # -------------------------------------------------------------------------
    #                           MODERN UI VERSION
    # -------------------------------------------------------------------------

    def _build_layout(self):
        # ---- Modern ttk styling ----
        style = ttk.Style(self.root)
        style.theme_use("clam")

        # Colors
        bg = "#ECECEC"
        frame_bg = "#F7F7F7"
        header_bg = "#3C3F41"
        header_fg = "#FFFFFF"
        accent = "#4A90E2"

        self.root.configure(bg=bg)

        style.configure(".", background=bg, font=("Segoe UI", 10))
        style.configure("TLabel", background=bg, foreground="#333")
        style.configure("TEntry", padding=4, fieldbackground="white")
        style.configure("TCombobox", padding=4)
        style.configure("TButton",
                        padding=4,
                        relief="flat",
                        background=accent,
                        foreground="white")

        # LabelFrame styling
        style.configure("TLabelframe", background=frame_bg, borderwidth=1, relief="solid")
        style.configure("TLabelframe.Label",
                        font=("Segoe UI Semibold", 11),
                        background=frame_bg,
                        foreground="#222")

        # Treeview styling
        style.configure("Treeview",
                        background="white",
                        foreground="#222",
                        rowheight=25,
                        fieldbackground="white",
                        bordercolor="#DDD",
                        borderwidth=1)

        style.configure("Treeview.Heading",
                        background=header_bg,
                        foreground=header_fg,
                        font=("Segoe UI Semibold", 10),
                        relief="flat")

        style.map("Treeview.Heading", background=[("active", "#505355")])

        PADY = 4

        # ---------- Filters Frame ----------
        filters_frame = ttk.LabelFrame(self.root, text="Filters")
        filters_frame.pack(fill="x", padx=12, pady=8)

        ttk.Label(filters_frame, text="Year min:").grid(row=0, column=0, sticky="w", padx=5, pady=PADY)
        self.combo_year_min = ttk.Combobox(filters_frame, width=12, state="readonly",
                                           values=[str(y) for y in range(1900, 2026)])
        self.combo_year_min.set("1990")
        self.combo_year_min.grid(row=0, column=1, padx=5, pady=PADY)

        ttk.Label(filters_frame, text="Year max:").grid(row=0, column=2, sticky="w", padx=5, pady=PADY)
        self.combo_year_max = ttk.Combobox(filters_frame, width=12, state="readonly",
                                           values=[str(y) for y in range(1900, 2026)])
        self.combo_year_max.set("2020")
        self.combo_year_max.grid(row=0, column=3, padx=5, pady=PADY)

        # Popularity
        pop_values = [f"{x/2:.1f}" for x in range(0, 41)]
        ttk.Label(filters_frame, text="Popularity min:").grid(row=1, column=0, sticky="w", padx=5, pady=PADY)
        self.combo_pop_min = ttk.Combobox(filters_frame, width=12, state="readonly", values=pop_values)
        self.combo_pop_min.set("0.0")
        self.combo_pop_min.grid(row=1, column=1, padx=5, pady=PADY)

        ttk.Label(filters_frame, text="Popularity max:").grid(row=1, column=2, sticky="w", padx=5, pady=PADY)
        self.combo_pop_max = ttk.Combobox(filters_frame, width=12, state="readonly", values=pop_values)
        self.combo_pop_max.set("20.0")
        self.combo_pop_max.grid(row=1, column=3, padx=5, pady=PADY)

        # Vote avg
        vote_values = [f"{x/2:.1f}" for x in range(0, 21)]
        ttk.Label(filters_frame, text="Vote avg min:").grid(row=2, column=0, sticky="w", padx=5, pady=PADY)
        self.combo_vote_min = ttk.Combobox(filters_frame, width=12, state="readonly", values=vote_values)
        self.combo_vote_min.set("3.0")
        self.combo_vote_min.grid(row=2, column=1, padx=5, pady=PADY)

        ttk.Label(filters_frame, text="Vote avg max:").grid(row=2, column=2, sticky="w", padx=5, pady=PADY)
        self.combo_vote_max = ttk.Combobox(filters_frame, width=12, state="readonly", values=vote_values)
        self.combo_vote_max.set("8.0")
        self.combo_vote_max.grid(row=2, column=3, padx=5, pady=PADY)

        # Runtime
        runtime_values = [str(v) for v in range(30, 241, 10)]
        ttk.Label(filters_frame, text="Runtime min:").grid(row=3, column=0, sticky="w", padx=5, pady=PADY)
        self.combo_runtime_min = ttk.Combobox(filters_frame, width=12, state="readonly", values=runtime_values)
        self.combo_runtime_min.set("30")
        self.combo_runtime_min.grid(row=3, column=1, padx=5, pady=PADY)

        ttk.Label(filters_frame, text="Runtime max:").grid(row=3, column=2, sticky="w", padx=5, pady=PADY)
        self.combo_runtime_max = ttk.Combobox(filters_frame, width=12, state="readonly", values=runtime_values)
        self.combo_runtime_max.set("180")
        self.combo_runtime_max.grid(row=3, column=3, padx=5, pady=PADY)

        # Countries
        ttk.Label(filters_frame, text="Countries (comma-separated):").grid(
            row=4, column=0, sticky="w", padx=5, pady=PADY)
        self.entry_countries = ttk.Entry(filters_frame, width=28)
        self.entry_countries.insert(0, "US,GB")
        self.entry_countries.grid(row=4, column=1, columnspan=3, sticky="we", padx=5, pady=PADY)

        # Language
        ttk.Label(filters_frame, text="Language:").grid(row=5, column=0, sticky="w", padx=5, pady=PADY)
        self.combo_language = ttk.Combobox(
            filters_frame, width=12, state="readonly",
            values=("en", "fr", "de", "es", "it", "ja", "ko", "zh")
        )
        self.combo_language.set("en")
        self.combo_language.grid(row=5, column=1, padx=5, pady=PADY)

        # Buttons
        self.btn_run_query = ttk.Button(filters_frame, text="Run query (show movies)",
                                        command=self.run_query)
        self.btn_run_query.grid(row=6, column=0, columnspan=2, pady=10, padx=5, sticky="we")

        self.btn_run_perf = ttk.Button(filters_frame,
                                       text="Run index + LSH performance (8 schemes)",
                                       command=self.run_index_performance)
        self.btn_run_perf.grid(row=6, column=2, columnspan=2, pady=10, padx=5, sticky="we")

        # ---------- Genre LSH search Frame (ΝΕΟ) ----------
        genre_frame = ttk.LabelFrame(self.root, text="Genre LSH search")
        genre_frame.pack(fill="x", padx=12, pady=8)

        ttk.Label(genre_frame, text="Select genre:").grid(row=0, column=0, sticky="w", padx=5, pady=PADY)
        self.combo_genres = ttk.Combobox(genre_frame, width=25, state="readonly", values=[])
        self.combo_genres.grid(row=0, column=1, padx=5, pady=PADY, sticky="w")

        self.btn_genre_search = ttk.Button(
            genre_frame,
            text="Search by Genre (LSH, top-5)",
            command=self.run_genre_search,
        )
        self.btn_genre_search.grid(row=0, column=2, padx=5, pady=PADY, sticky="we")

        # ---------- Status ----------
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", padx=12, pady=4)
        ttk.Label(status_frame, textvariable=self.status_var,
                  font=("Segoe UI", 10, "italic")).pack(side="left")

        # ---------- Results table (movies) ----------
        results_frame = ttk.LabelFrame(self.root, text="Query results (movies)")
        results_frame.pack(fill="x", expand=False, padx=12, pady=8, ipady=5)

        columns = ["title", "release_year", "popularity", "vote_average", "runtime"]
        self.tree_movies = ttk.Treeview(results_frame, columns=columns, show="headings", height=6)

        for col in columns:
            self.tree_movies.heading(col, text=col,
                                     command=lambda c=col: treeview_sort_column(self.tree_movies, c, False))
            self.tree_movies.column(col, width=140, anchor="w")

        vsb = ttk.Scrollbar(results_frame, orient="vertical", command=self.tree_movies.yview)
        hsb = ttk.Scrollbar(results_frame, orient="horizontal", command=self.tree_movies.xview)
        self.tree_movies.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree_movies.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)

        # ---------- Numeric index performance table ----------
        perf_numeric_frame = ttk.LabelFrame(self.root, text="Numeric index performance")
        perf_numeric_frame.pack(fill="both", expand=False, padx=12, pady=8)

        perf_num_columns = ["Index", "Build_time_s", "Numeric_time_s", "Numeric_cand"]
        self.tree_perf_numeric = ttk.Treeview(perf_numeric_frame,
                                              columns=perf_num_columns,
                                              show="headings", height=5)

        for col in perf_num_columns:
            self.tree_perf_numeric.heading(col, text=col,
                                           command=lambda c=col: treeview_sort_column(self.tree_perf_numeric, c, False))
            self.tree_perf_numeric.column(col, width=140, anchor="center")

        vsb_num = ttk.Scrollbar(perf_numeric_frame, orient="vertical",
                                command=self.tree_perf_numeric.yview)
        self.tree_perf_numeric.configure(yscrollcommand=vsb_num.set)

        self.tree_perf_numeric.grid(row=0, column=0, sticky="nsew")
        vsb_num.grid(row=0, column=1, sticky="ns")

        perf_numeric_frame.rowconfigure(0, weight=1)
        perf_numeric_frame.columnconfigure(0, weight=1)

        # ---------- Schemes (Index + LSH) performance ----------
        perf_scheme_frame = ttk.LabelFrame(self.root, text="Schemes: (Index + LSH) performance")
        perf_scheme_frame.pack(fill="both", expand=False, padx=12, pady=8)

        perf_scheme_columns = ["Scheme", "IdxBuild_s", "NumQ_s", "Cand",
                               "Res", "LSH_build_s", "LSH_query_s"]

        self.tree_perf_scheme = ttk.Treeview(perf_scheme_frame,
                                             columns=perf_scheme_columns,
                                             show="headings", height=6)

        for col in perf_scheme_columns:
            self.tree_perf_scheme.heading(col, text=col,
                                          command=lambda c=col: treeview_sort_column(self.tree_perf_scheme, c, False))
            self.tree_perf_scheme.column(col, width=130, anchor="center")

        vsb_scheme = ttk.Scrollbar(perf_scheme_frame, orient="vertical",
                                   command=self.tree_perf_scheme.yview)
        self.tree_perf_scheme.configure(yscrollcommand=vsb_scheme.set)

        self.tree_perf_scheme.grid(row=0, column=0, sticky="nsew")
        vsb_scheme.grid(row=0, column=1, sticky="ns")

        perf_scheme_frame.rowconfigure(0, weight=1)
        perf_scheme_frame.columnconfigure(0, weight=1)

        # ---------- LSH Info ----------
        lsh_frame = ttk.Frame(self.root, padding=6)
        lsh_frame.pack(fill="x", padx=12, pady=4)
        lsh_frame.configure(style="TLabelframe")
        ttk.Label(lsh_frame, textvariable=self.lsh_common_var,
                  font=("Segoe UI", 10, "italic")).pack(side="left")

    # -------------------------------------------------------------------------
    #                          DATA HELPERS
    # -------------------------------------------------------------------------

    def ensure_base_df_loaded(self):
        if self.base_df is not None:
            return

        try:
            self.status_var.set("Loading dataset (this may take a bit)...")
            self.root.update_idletasks()

            df_raw = load_dataset()
            df_processed = preprocess_dataset(df_raw)
            self.base_df = build_base_pool(df_processed)

            self.status_var.set(f"Base pool loaded: {len(self.base_df)} movies.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
            self.status_var.set("Error loading data.")
            self.base_df = None

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
            messagebox.showerror("Error", "Some dropdown values are invalid.")
            return None

        countries_text = self.entry_countries.get().strip()
        if countries_text:
            countries = tuple(c.strip().upper()
                              for c in countries_text.split(",") if c.strip())
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

    # -------------------------------------------------------------------------
    #                               QUERY
    # -------------------------------------------------------------------------

    def run_query(self):
        self.ensure_base_df_loaded()
        if self.base_df is None:
            return

        params = self.get_params_from_ui()
        if params is None:
            return

        self.status_var.set("Running query (pure pandas)...")
        self.root.update_idletasks()

        result_df = apply_full_query_filters(self.base_df, params)
        self.last_ground_truth_df = result_df

        # Με κάθε νέο query, αδειάζουμε genre index για να ξαναχτιστεί στο νέο subset
        self.genre_index = None
        self.all_genres = None
        self.combo_genres["values"] = []
        self.combo_genres.set("")

        self.status_var.set(f"Query done. Found {len(result_df)} movies.")
        self.update_movies_table(result_df)

    def update_movies_table(self, df: pd.DataFrame):
        for row in self.tree_movies.get_children():
            self.tree_movies.delete(row)

        if df.empty:
            return

        max_rows = 300
        view_df = df[["title", "release_year", "popularity",
                      "vote_average", "runtime"]].head(max_rows)

        for _, row in view_df.iterrows():
            values = [
                str(row["title"]),
                str(row["release_year"]),
                f"{row['popularity']:.2f}",
                f"{row['vote_average']:.1f}",
                str(int(row["runtime"])),
            ]
            self.tree_movies.insert("", "end", values=values)

    # -------------------------------------------------------------------------
    #                     GENRE LSH SEARCH (ΝΕΟ)
    # -------------------------------------------------------------------------

    def ensure_genre_index_built(self) -> bool:
        """
        Φροντίζει να υπάρχει last_ground_truth_df από το query,
        να φτιαχτεί ένα GenreLSHIndex πάνω σε αυτό, και να γεμίσει το combobox με genres.
        """
        self.ensure_base_df_loaded()
        if self.base_df is None:
            return False

        if self.last_ground_truth_df is None or self.last_ground_truth_df.empty:
            messagebox.showinfo(
                "Info",
                "Please run a numeric query first (Run query) to define a movie subset."
            )
            return False

        if self.genre_index is not None and self.all_genres:
            # Ήδη χτισμένο
            return True

        df = self.last_ground_truth_df

        all_genres = extract_all_genres(df)
        if not all_genres:
            messagebox.showinfo(
                "Info",
                "No genres found in the current query result."
            )
            return False

        self.all_genres = all_genres
        self.genre_index = build_genre_lsh_index(df)

        # Γέμισε το combobox
        self.combo_genres["values"] = self.all_genres
        if not self.combo_genres.get() and self.all_genres:
            self.combo_genres.set(self.all_genres[0])

        return True

    def run_genre_search(self):
        """
        Ο χρήστης επιλέγει ένα genre από το combobox και
        βρίσκουμε top-5 ταινίες με βάση LSH στα genres.
        """
        if not self.ensure_genre_index_built():
            return

        selected_genre = self.combo_genres.get().strip()
        if not selected_genre:
            messagebox.showwarning("Warning", "Please select a genre first.")
            return

        self.status_var.set(f"Running LSH search for genre: {selected_genre} ...")
        self.root.update_idletasks()

        assert self.genre_index is not None
        result_df = query_by_genre_name(self.genre_index, selected_genre, top_n=5)

        if result_df.empty:
            self.status_var.set(f"No similar movies found for genre '{selected_genre}'.")
            self.update_movies_table(result_df)
            return

        self.status_var.set(
            f"Genre LSH search done. Found {len(result_df)} movies (top-5 displayed)."
        )
        self.update_movies_table(result_df)

    # -------------------------------------------------------------------------
    #                Index + LSH Performance (όπως πριν)
    # -------------------------------------------------------------------------

    def run_index_performance(self):
        self.ensure_base_df_loaded()
        if self.base_df is None:
            return

        params = self.get_params_from_ui()
        if params is None:
            return

        self.status_var.set("Running index + LSH performance...")
        self.root.update_idletasks()

        ground_truth_df = apply_full_query_filters(self.base_df, params)
        self.last_ground_truth_df = ground_truth_df

        # ----- NUMERIC INDEX PERFORMANCE -----
        numeric_summary = []

        kd_features = ["release_year", "popularity",
                       "vote_average", "runtime", "vote_count"]

        kd_tree, kd_build_time = build_kd_index(self.base_df, kd_features)
        kd_res, kd_num_time = numeric_candidates_kdtree(
            kd_tree, self.base_df, params, kd_features)
        numeric_summary.append(("KD-Tree", kd_build_time, kd_num_time, len(kd_res)))

        quad_tree, quad_build_time = build_quad_index(self.base_df)
        quad_res, quad_num_time = numeric_candidates_quadtree(
            quad_tree, self.base_df, params)
        numeric_summary.append(("Quad-Tree", quad_build_time, quad_num_time, len(quad_res)))

        year_tree, range_build_time = build_range_index(self.base_df)
        range_res, range_num_time = numeric_candidates_rangetree(
            year_tree, self.base_df, params)
        numeric_summary.append(("Range-Tree", range_build_time,
                               range_num_time, len(range_res)))

        rtree_obj, rtree_build_time = build_rtree_index(self.base_df)
        rtree_res, rtree_num_time = numeric_candidates_rtree(
            rtree_obj, self.base_df, params)
        numeric_summary.append(("R-Tree", rtree_build_time, rtree_num_time, len(rtree_res)))

        self.update_numeric_perf_table(numeric_summary)

        # ----- SCHEMES (Index + LSH) -----
        scheme_summary = []

        # KD
        kd_scheme = apply_full_query_filters(kd_res, params)
        kd_lsh_b, kd_lsh_q = run_lsh_on_df(kd_scheme, "KD-Tree + LSH", N=3)
        scheme_summary.append(("KD-Tree + LSH", kd_build_time, kd_num_time,
                               len(kd_res), len(kd_scheme), kd_lsh_b, kd_lsh_q))

        # Quad
        quad_scheme = apply_full_query_filters(quad_res, params)
        quad_lsh_b, quad_lsh_q = run_lsh_on_df(quad_scheme, "Quad-Tree + LSH", N=3)
        scheme_summary.append(("Quad-Tree + LSH", quad_build_time, quad_num_time,
                               len(quad_res), len(quad_scheme),
                               quad_lsh_b, quad_lsh_q))

        # Range
        range_scheme = apply_full_query_filters(range_res, params)
        range_lsh_b, range_lsh_q = run_lsh_on_df(range_scheme, "Range-Tree + LSH", N=3)
        scheme_summary.append(("Range-Tree + LSH", range_build_time, range_num_time,
                               len(range_res), len(range_scheme),
                               range_lsh_b, range_lsh_q))

        # R-Tree
        rtree_scheme = apply_full_query_filters(rtree_res, params)
        rtree_lsh_b, rtree_lsh_q = run_lsh_on_df(rtree_scheme, "R-Tree + LSH", N=3)
        scheme_summary.append(("R-Tree + LSH", rtree_build_time, rtree_num_time,
                               len(rtree_res), len(rtree_scheme),
                               rtree_lsh_b, rtree_lsh_q))

        self.update_scheme_perf_table(scheme_summary)

        # Common LSH
        lsh_common_b, lsh_common_q = run_lsh_on_df(ground_truth_df, "COMMON", N=3)
        self.lsh_common_var.set(
            f"Common LSH on GT (|GT|={len(ground_truth_df)}): "
            f"build={lsh_common_b:.4f}s, query={lsh_common_q:.6f}s"
        )

        self.status_var.set("Index + LSH performance done.")

    # ---------- TABLE UPDATES ----------
    def update_numeric_perf_table(self, rows):
        for row in self.tree_perf_numeric.get_children():
            self.tree_perf_numeric.delete(row)

        for name, build, num, cand in rows:
            self.tree_perf_numeric.insert(
                "", "end",
                values=[name, f"{build:.4f}", f"{num:.4f}", str(cand)]
            )

    def update_scheme_perf_table(self, rows):
        for row in self.tree_perf_scheme.get_children():
            self.tree_perf_scheme.delete(row)

        for scheme, bld, num, cand, res, lsh_b, lsh_q in rows:
            self.tree_perf_scheme.insert(
                "", "end",
                values=[scheme,
                        f"{bld:.4f}",
                        f"{num:.4f}",
                        str(cand),
                        str(res),
                        f"{lsh_b:.4f}",
                        f"{lsh_q:.6f}"]
            )


def main():
    root = tk.Tk()
    app = MovieQueryApp(root)
    root.geometry("1100x980")
    root.mainloop()


if __name__ == "__main__":
    main()
