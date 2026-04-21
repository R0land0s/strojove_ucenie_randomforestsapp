import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import tkinter as tk
from tkinter import ttk, messagebox, font
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_curve, auc,
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score
)

BG       = "#0d0f14"
PANEL    = "#141720"
CARD     = "#1c2030"
ACCENT   = "#e63946"
ACCENT2  = "#457b9d"
TEXT     = "#eef0f4"
MUTED    = "#6b7280"
GREEN    = "#2dc653"
ORANGE   = "#f4a261"
PURPLE   = "#9b59b6"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    MUTED,
    "axes.labelcolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT,
    "grid.color":        "#2a2d3a",
    "grid.alpha":        0.5,
    "font.family":       "monospace",
})


# DATA LOADING & PREPROCESSING

DATASET_PATH = r"C:\Users\Roland\OneDrive\Počítač\heart.csv"

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Encode categoricals
    cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    encoders = {}
    df_enc = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        encoders[col] = le

    feature_cols = [c for c in df_enc.columns if c != "HeartDisease"]
    X = df_enc[feature_cols].values
    y = df_enc["HeartDisease"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return df, df_enc, X_train, X_test, y_train, y_test, feature_cols, encoders


def train_model(X_train, y_train, n_estimators=200, max_depth=None, min_samples_split=2):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth if max_depth != 0 else None,
        min_samples_split=min_samples_split,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    return clf


class HeartApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("❤  Heart Disease · Random Forest Classifier")
        self.configure(bg=BG)
        self.state("zoomed")           # start maximised (Windows)
        self.minsize(1200, 750)

        # ── load data ──────────────────────────
        try:
            (self.df, self.df_enc,
             self.X_train, self.X_test,
             self.y_train, self.y_test,
             self.feature_cols, self.encoders) = load_and_preprocess(DATASET_PATH)
        except FileNotFoundError:
            messagebox.showerror(
                "Chyba",
                f"Dataset nebol nájdený:\n{DATASET_PATH}\n\n"
                "Skontroluj cestu v premennej DATASET_PATH."
            )
            self.destroy()
            return

        self.model = None

        self._build_ui()
        self._train_and_refresh()

    def _build_ui(self):
        # ── header ──
        hdr = tk.Frame(self, bg=ACCENT, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(
            hdr,
            text="  ♥  HEART DISEASE  ·  RANDOM FOREST CLASSIFIER",
            bg=ACCENT, fg=TEXT,
            font=("Courier New", 15, "bold"),
            anchor="w"
        ).pack(side="left", padx=20)
        tk.Label(
            hdr,
            text=f"dataset: {len(self.df)} obs  ·  {len(self.feature_cols)} features",
            bg=ACCENT, fg="#ffd6d6",
            font=("Courier New", 10),
        ).pack(side="right", padx=20)

        # ── main body: left controls + right charts ──
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg=PANEL, width=280)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self._build_left_panel(left)
        self._build_notebook(right)

    # VYSVETLIVKY ATRIBÚTOV

    def _build_info_tab(self):
        parent = self.tab_info

        ATTRS = [
            ("1", "Age", "Vek pacienta",
             "Numerická hodnota v rokoch.",
             []),
            ("2", "Sex", "Pohlavie pacienta",
             "Kategorická premenná.",
             [("M", "Muž (Male)"), ("F", "Žena (Female)")]),
            ("3", "ChestPainType", "Typ bolesti na hrudi",
             "Kategorická premenná – typ bolesti.",
             [("TA",  "Typická angína (Typical Angina)"),
              ("ATA", "Atypická angína (Atypical Angina)"),
              ("NAP", "Neangínová bolesť (Non-Anginal Pain)"),
              ("ASY", "Asymptomatický (Asymptomatic)")]),
            ("4", "RestingBP", "Krvný tlak v pokoji",
             "Nameraný pri prijatí do nemocnice [mm Hg].",
             []),
            ("5", "Cholesterol", "Sérový cholesterol",
             "Nameraná hodnota cholesterolu v krvi [mm/dl].",
             []),
            ("6", "FastingBS", "Glykémia nalačno",
             "Hladina cukru v krvi nalačno.",
             [("1", "Glykémia > 120 mg/dl"),
              ("0", "Glykémia ≤ 120 mg/dl")]),
            ("7", "RestingECG", "EKG v pokoji",
             "Výsledky elektrokardiogramu v pokoji.",
             [("Normal", "Normálny nález"),
              ("ST",     "Abnormalita ST-T vlny (inverzie T vlny alebo elevácia/depresia ST > 0,05 mV)"),
              ("LVH",    "Pravdepodobná alebo definitívna ľavostranná hypertrofia komory (Estesove kritériá)")]),
            ("8", "MaxHR", "Maximálna tepová frekvencia",
             "Najvyššia dosiahnutá TF počas záťažového testu. Numerická hodnota medzi 60 a 202.",
             []),
            ("9", "ExerciseAngina", "Angína vyvolaná záťažou",
             "Či sa počas cvičenia objavila angína.",
             [("Y", "Áno (Yes)"), ("N", "Nie (No)")]),
            ("10", "Oldpeak", "Oldpeak (ST depresia)",
             "Hodnota ST depresie vyvolanej záťažou oproti pokoju. Číselná hodnota.",
             []),
            ("11", "ST_Slope", "Sklon ST segmentu",
             "Sklon vrcholu ST segmentu počas záťažového testu.",
             [("Up",   "Vzostupný (Upsloping)"),
              ("Flat", "Plochý (Flat)"),
              ("Down", "Zostupný (Downsloping)")]),
            ("12", "HeartDisease", "Cieľová premenná – srdcová choroba",
             "Výstupná trieda modelu.",
             [("1", "Srdcová choroba prítomná"),
              ("0", "Normálny nález")]),
        ]

        # scrollable canvas
        canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=BG)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_resize(e):
            canvas.itemconfig(win_id, width=e.width)
        canvas.bind("<Configure>", _on_resize)

        def _on_frame_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", _on_frame_configure)

        # mousewheel scroll
        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ── nadpis ──
        tk.Label(
            inner,
            text="POPIS ATRIBÚTOV DATASETU",
            bg=BG, fg=ACCENT,
            font=("Courier New", 14, "bold"),
            anchor="w"
        ).pack(fill="x", padx=28, pady=(22, 2))
        tk.Label(
            inner,
            text="Heart Failure Prediction Dataset  ·  918 pozorovaní  ·  11 príznakov + 1 cieľová premenná",
            bg=BG, fg=MUTED,
            font=("Courier New", 9),
            anchor="w"
        ).pack(fill="x", padx=28, pady=(0, 18))

        accent_colors = [ACCENT, ACCENT2, GREEN, ORANGE, PURPLE, "#f1c40f",
                         ACCENT, ACCENT2, GREEN, ORANGE, PURPLE, ACCENT]

        for i, (num, name, sk_name, desc, vals) in enumerate(ATTRS):
            color = accent_colors[i % len(accent_colors)]

            # card frame
            card = tk.Frame(inner, bg=CARD, bd=0)
            card.pack(fill="x", padx=24, pady=6, ipady=10)

            # left colour stripe
            stripe = tk.Frame(card, bg=color, width=5)
            stripe.pack(side="left", fill="y")

            body_frame = tk.Frame(card, bg=CARD)
            body_frame.pack(side="left", fill="both", expand=True, padx=14, pady=6)

            # header row
            hdr_row = tk.Frame(body_frame, bg=CARD)
            hdr_row.pack(fill="x")
            tk.Label(
                hdr_row,
                text=f"#{num}",
                bg=CARD, fg=color,
                font=("Courier New", 10, "bold"),
                width=4, anchor="w"
            ).pack(side="left")
            tk.Label(
                hdr_row,
                text=name,
                bg=CARD, fg=TEXT,
                font=("Courier New", 12, "bold"),
                anchor="w"
            ).pack(side="left")
            tk.Label(
                hdr_row,
                text=f"  –  {sk_name}",
                bg=CARD, fg=MUTED,
                font=("Courier New", 10),
                anchor="w"
            ).pack(side="left")

            # description
            tk.Label(
                body_frame,
                text=desc,
                bg=CARD, fg=MUTED,
                font=("Courier New", 9),
                anchor="w", justify="left"
            ).pack(fill="x", pady=(2, 4))

            # values grid
            if vals:
                vals_frame = tk.Frame(body_frame, bg=CARD)
                vals_frame.pack(fill="x")
                for code, meaning in vals:
                    vrow = tk.Frame(vals_frame, bg=CARD)
                    vrow.pack(anchor="w", pady=1)
                    tk.Label(
                        vrow,
                        text=f"  {code}",
                        bg=CARD, fg=color,
                        font=("Courier New", 9, "bold"),
                        width=8, anchor="w"
                    ).pack(side="left")
                    tk.Label(
                        vrow,
                        text=f"→  {meaning}",
                        bg=CARD, fg=TEXT,
                        font=("Courier New", 9),
                        anchor="w"
                    ).pack(side="left")

        # bottom padding
        tk.Frame(inner, bg=BG, height=20).pack()

    # LEFT PANEL  – hyperparams + custom input
    def _build_left_panel(self, parent):
        def section(text):
            tk.Label(
                parent, text=text.upper(),
                bg=PANEL, fg=ACCENT,
                font=("Courier New", 9, "bold"),
                anchor="w"
            ).pack(fill="x", padx=14, pady=(18, 4))

        def hline():
            tk.Frame(parent, bg=MUTED, height=1).pack(fill="x", padx=14, pady=2)

        # ── model hyperparams ──────────────────
        section("▸ Model Hyperparametre")

        self._slider("Počet stromov (n_estimators)", "n_est", 50, 500, 200, parent)
        self._slider("Max hĺbka (0 = None)", "max_dep", 0, 30, 0, parent)
        self._slider("Min vzoriek na split", "min_split", 2, 20, 2, parent)

        tk.Button(
            parent,
            text="⟳  RETRÉNOVAŤ MODEL",
            command=self._train_and_refresh,
            bg=ACCENT, fg=TEXT,
            font=("Courier New", 10, "bold"),
            relief="flat", cursor="hand2",
            activebackground="#c1121f",
            pady=8
        ).pack(fill="x", padx=14, pady=(14, 4))

        hline()

        # ── custom prediction ──────────────────
        section("▸ Vlastný Pacient – Predikcia")

        fields_cfg = [
            ("Vek",              "Age",            "num",  None),
            ("Pohlavie",         "Sex",            "cat",  ["M", "F"]),
            ("Typ bolesti hrude","ChestPainType",  "cat",  ["ATA","NAP","ASY","TA"]),
            ("Tlak v pokoji",    "RestingBP",      "num",  None),
            ("Cholesterol",      "Cholesterol",    "num",  None),
            ("Glykémia > 120",   "FastingBS",      "cat",  ["0","1"]),
            ("RestingECG",       "RestingECG",     "cat",  ["Normal","ST","LVH"]),
            ("Max TF",           "MaxHR",          "num",  None),
            ("Angina pri cvič.", "ExerciseAngina", "cat",  ["N","Y"]),
            ("Oldpeak",          "Oldpeak",        "num",  None),
            ("ST sklon",         "ST_Slope",       "cat",  ["Up","Flat","Down"]),
        ]

        self.input_vars = {}
        for label, key, kind, opts in fields_cfg:
            row = tk.Frame(parent, bg=PANEL)
            row.pack(fill="x", padx=14, pady=2)
            tk.Label(row, text=label, bg=PANEL, fg=MUTED,
                     font=("Courier New", 8), anchor="w", width=18).pack(side="left")
            if kind == "num":
                var = tk.StringVar()
                e = tk.Entry(row, textvariable=var, width=8,
                             bg=CARD, fg=TEXT, insertbackground=TEXT,
                             relief="flat", font=("Courier New", 9))
                e.pack(side="right")
                self.input_vars[key] = var
            else:
                var = tk.StringVar(value=opts[0])
                cb = ttk.Combobox(row, textvariable=var, values=opts,
                                  width=7, state="readonly",
                                  font=("Courier New", 9))
                cb.pack(side="right")
                self.input_vars[key] = var

        # style combobox – zosúladiť s pozadím panelu
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox",
                        fieldbackground=PANEL, background=PANEL,
                        foreground=TEXT, selectbackground=PANEL,
                        selectforeground=TEXT,
                        bordercolor=PANEL, lightcolor=PANEL,
                        darkcolor=PANEL, arrowcolor=MUTED,
                        relief="flat")
        style.map("TCombobox",
                  fieldbackground=[("readonly", PANEL), ("active", PANEL)],
                  background=[("readonly", PANEL), ("active", PANEL)],
                  bordercolor=[("readonly", PANEL), ("focus", MUTED)],
                  lightcolor=[("readonly", PANEL)],
                  darkcolor=[("readonly", PANEL)])

        tk.Button(
            parent,
            text="🔍  KLASIFIKOVAŤ PACIENTA",
            command=self._predict_custom,
            bg=ACCENT2, fg=TEXT,
            font=("Courier New", 10, "bold"),
            relief="flat", cursor="hand2",
            activebackground="#1d6083",
            pady=8
        ).pack(fill="x", padx=14, pady=(12, 4))

        self.result_lbl = tk.Label(
            parent, text="", bg=PANEL,
            font=("Courier New", 12, "bold"),
            wraplength=240
        )
        self.result_lbl.pack(padx=14, pady=6)

    def _slider(self, label, key, lo, hi, default, parent):
        var = tk.IntVar(value=default)
        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill="x", padx=14, pady=2)
        tk.Label(f, text=label, bg=PANEL, fg=MUTED,
                 font=("Courier New", 8), anchor="w").pack(fill="x")
        row = tk.Frame(f, bg=PANEL)
        row.pack(fill="x")
        val_lbl = tk.Label(row, textvariable=var, bg=PANEL, fg=ACCENT,
                            font=("Courier New", 9, "bold"), width=4)
        val_lbl.pack(side="right")
        tk.Scale(
            row, variable=var, from_=lo, to=hi,
            orient="horizontal", bg=PANEL, fg=TEXT,
            troughcolor=CARD, activebackground=ACCENT,
            highlightthickness=0, showvalue=False, bd=0
        ).pack(side="left", fill="x", expand=True)
        setattr(self, f"var_{key}", var)

    # RIGHT PANEL – notebook with tabs
    def _build_notebook(self, parent):
        style = ttk.Style()
        style.configure("Dark.TNotebook",             background=BG, borderwidth=0)
        style.configure("Dark.TNotebook.Tab",
                        background=PANEL, foreground=MUTED,
                        font=("Courier New", 10), padding=[14, 6])
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", CARD)],
                  foreground=[("selected", ACCENT)])

        nb = ttk.Notebook(parent, style="Dark.TNotebook")
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        # tabs
        self.tab_roc      = tk.Frame(nb, bg=BG)
        self.tab_metrics  = tk.Frame(nb, bg=BG)
        self.tab_feat     = tk.Frame(nb, bg=BG)
        self.tab_matrix   = tk.Frame(nb, bg=BG)
        self.tab_info     = tk.Frame(nb, bg=BG)

        nb.add(self.tab_roc,     text="  ROC Krivka  ")
        nb.add(self.tab_metrics, text="  Metriky modelu  ")
        nb.add(self.tab_feat,    text="  Dôležitosť príznakov  ")
        nb.add(self.tab_matrix,  text="  Konfúzna matica  ")
        nb.add(self.tab_info,    text="  Vysvetlivky atribútov  ")

        self._build_info_tab()

    def _train_and_refresh(self):
        n_est     = self.var_n_est.get()
        max_dep   = self.var_max_dep.get()
        min_split = self.var_min_split.get()

        self.model = train_model(
            self.X_train, self.y_train,
            n_estimators=n_est,
            max_depth=max_dep,
            min_samples_split=min_split
        )
        self.y_pred      = self.model.predict(self.X_test)
        self.y_prob      = self.model.predict_proba(self.X_test)[:, 1]
        self.fpr, self.tpr, _ = roc_curve(self.y_test, self.y_prob)
        self.roc_auc     = auc(self.fpr, self.tpr)

        self._draw_roc()
        self._draw_metrics()
        self._draw_features()
        self._draw_matrix()

    def _draw_roc(self):
        for w in self.tab_roc.winfo_children():
            w.destroy()

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(PANEL)

        # shaded area under ROC
        ax.fill_between(self.fpr, self.tpr, alpha=0.18, color=ACCENT)
        ax.plot(self.fpr, self.tpr, color=ACCENT, lw=2.5,
                label=f"ROC (AUC = {self.roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2, linestyle="--", label="Náhodný model")

        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
        ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
        ax.set_title("ROC Krivka – Random Forest", fontsize=13, fontweight="bold", color=ACCENT)
        ax.legend(loc="lower right", facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT, fontsize=10)
        ax.grid(True, alpha=0.3)

        # annotate AUC
        ax.annotate(f"AUC = {self.roc_auc:.4f}",
                    xy=(0.6, 0.3), xycoords="axes fraction",
                    fontsize=16, color=ACCENT, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", fc=CARD, ec=ACCENT, alpha=0.9))

        canvas = FigureCanvasTkAgg(fig, master=self.tab_roc)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    def _draw_metrics(self):
        for w in self.tab_metrics.winfo_children():
            w.destroy()

        acc      = accuracy_score(self.y_test, self.y_pred)
        bal_acc  = balanced_accuracy_score(self.y_test, self.y_pred)
        prec     = precision_score(self.y_test, self.y_pred)
        rec      = recall_score(self.y_test, self.y_pred)
        f1       = f1_score(self.y_test, self.y_pred)
        mcc      = matthews_corrcoef(self.y_test, self.y_pred)

        # cross-val
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, 
                                    np.vstack([self.X_train, self.X_test]),
                                    np.hstack([self.y_train, self.y_test]),
                                    cv=cv, scoring="roc_auc")

        fig = plt.figure(figsize=(10, 5))
        fig.patch.set_facecolor(BG)
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

        # ── bar chart of metrics ──
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor(PANEL)
        labels  = ["Accuracy", "Bal.Acc.", "Precision", "Recall", "F1", "MCC", "AUC"]
        values  = [acc, bal_acc, prec, rec, f1, (mcc+1)/2, self.roc_auc]
        colors  = [ACCENT2, PURPLE, GREEN, ORANGE, ACCENT, "#f1c40f", ACCENT]
        bars = ax1.barh(labels, values, color=colors, height=0.55, edgecolor="none")
        for bar, val, raw in zip(bars, values, [acc, bal_acc, prec, rec, f1, mcc, self.roc_auc]):
            ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                     f"{raw:.4f}", va="center", ha="left", fontsize=9.5,
                     color=TEXT, fontweight="bold")
        ax1.set_xlim(0, 1.18)
        ax1.set_title("Výkonnostné metriky", fontsize=12, fontweight="bold", color=ACCENT)
        ax1.axvline(0.5, color=MUTED, lw=0.8, ls="--")
        ax1.grid(axis="x", alpha=0.3)
        ax1.tick_params(axis="y", labelsize=10)
        note = tk.Label(self.tab_metrics, text="MCC škálovaný do [0,1] pre zobrazenie; skutočná hodnota je v texte.",
                        bg=BG, fg=MUTED, font=("Courier New", 7))

        # ── cross-val boxplot ──
        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor(PANEL)
        bp = ax2.boxplot(cv_scores, patch_artist=True, widths=0.4,
                         medianprops=dict(color=ACCENT, lw=2.5),
                         boxprops=dict(facecolor=CARD, color=ACCENT2),
                         whiskerprops=dict(color=ACCENT2),
                         capprops=dict(color=ACCENT2),
                         flierprops=dict(marker="o", color=ACCENT, markersize=5))
        ax2.scatter([1]*len(cv_scores), cv_scores, color=ORANGE, zorder=5,
                    s=50, alpha=0.85, label="Fold AUC")
        ax2.set_xticklabels(["5-fold CV AUC"])
        ax2.set_ylim(0.5, 1.05)
        ax2.set_title("Krížová validácia (AUC)", fontsize=12, fontweight="bold", color=ACCENT)
        ax2.legend(facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT, fontsize=9)
        ax2.grid(axis="y", alpha=0.3)
        mean_cv = cv_scores.mean()
        ax2.annotate(f"μ = {mean_cv:.4f}\nσ = {cv_scores.std():.4f}",
                     xy=(0.62, 0.12), xycoords="axes fraction",
                     fontsize=10, color=TEXT,
                     bbox=dict(boxstyle="round", fc=CARD, ec=ACCENT2))

        canvas = FigureCanvasTkAgg(fig, master=self.tab_metrics)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # text summary below
        report = classification_report(self.y_test, self.y_pred,
                                       target_names=["Zdravý (0)", "Chorý  (1)"])
        txt = tk.Text(self.tab_metrics, bg=CARD, fg=TEXT, height=9,
                      font=("Courier New", 9), relief="flat", bd=0)
        txt.insert("1.0", report)
        txt.config(state="disabled")
        txt.pack(fill="x", padx=8, pady=(0, 4))
        note.pack()
        plt.close(fig)

    def _draw_features(self):
        for w in self.tab_feat.winfo_children():
            w.destroy()

        importances = self.model.feature_importances_
        idx = np.argsort(importances)

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(PANEL)

        palette = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(idx)))
        bars = ax.barh(np.array(self.feature_cols)[idx], importances[idx],
                       color=palette, edgecolor="none", height=0.6)
        for bar, val in zip(bars, importances[idx]):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=8.5, color=TEXT)

        ax.set_xlabel("Gini dôležitosť", fontsize=11)
        ax.set_title("Dôležitosť príznakov (Feature Importance)", fontsize=13,
                     fontweight="bold", color=ACCENT)
        ax.grid(axis="x", alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=self.tab_feat)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    def _draw_matrix(self):
        for w in self.tab_matrix.winfo_children():
            w.destroy()

        cm = confusion_matrix(self.y_test, self.y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(PANEL)

        im = ax.imshow(cm, cmap="Reds", aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        fontsize=22, fontweight="bold",
                        color=TEXT if cm[i, j] < cm.max()*0.6 else BG)

        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Zdravý", "Pred: Chorý"], fontsize=10)
        ax.set_yticklabels(["Skutočný: Zdravý", "Skutočný: Chorý"], fontsize=10)
        ax.set_title("Konfúzna matica", fontsize=13, fontweight="bold", color=ACCENT)
        plt.colorbar(im, ax=ax)

        tn, fp, fn, tp = cm.ravel()
        info = (f"  TN (True Neg)  = {tn}   FP (False Pos) = {fp}\n"
                f"  FN (False Neg) = {fn}   TP (True Pos)  = {tp}\n\n"
                f"  Sensitivity (Recall) = {tp/(tp+fn):.4f}   "
                f"Specificity = {tn/(tn+fp):.4f}")

        canvas = FigureCanvasTkAgg(fig, master=self.tab_matrix)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        tk.Label(self.tab_matrix, text=info,
                 bg=BG, fg=TEXT, font=("Courier New", 10),
                 justify="left").pack(anchor="w", padx=20, pady=8)
        plt.close(fig)

    def _predict_custom(self):
        if self.model is None:
            messagebox.showwarning("Model", "Model ešte nie je natrénovaný.")
            return

        try:
            row = []
            for col in self.feature_cols:
                raw = self.input_vars[col].get().strip()
                if col in self.encoders:
                    enc_val = self.encoders[col].transform([raw])[0]
                    row.append(enc_val)
                else:
                    row.append(float(raw))

            X_new    = np.array(row).reshape(1, -1)
            pred     = self.model.predict(X_new)[0]
            prob     = self.model.predict_proba(X_new)[0]

            if pred == 1:
                label = "⚠  CHORÝ  (Heart Disease)"
                color = ACCENT
            else:
                label = "✓  ZDRAVÝ  (No Heart Disease)"
                color = GREEN

            msg = f"{label}\n\nPravdepodobnosť choroby: {prob[1]*100:.1f} %"
            self.result_lbl.config(text=msg, fg=color)

        except ValueError as e:
            messagebox.showerror("Chyba vstupu", f"Skontroluj číselné polia!\n{e}")
        except Exception as e:
            messagebox.showerror("Chyba", str(e))


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = HeartApp()
    app.mainloop()