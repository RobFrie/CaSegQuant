import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from zipfile import ZipFile
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from filterpy.kalman import KalmanFilter
from scipy.stats import wilcoxon
from sklearn.metrics import mean_squared_error
import os
import sys

# Funktionen für Filtermethoden
def binomial_filter(series):
    # Binomial-Filter wird auf die Serie angewendet
    filtered_series = series.copy()
    for t in range(1, len(series) - 1):
        filtered_series.iloc[t] = 0.25 * series.iloc[t - 1] + 0.5 * series.iloc[t] + 0.25 * series.iloc[t + 1]
    return filtered_series

def gaussian_smoothing(series, window_size, sigma):
    # Gauss-Glättung wird auf die Serie angewendet
    return gaussian_filter1d(series, sigma=sigma, mode='reflect')

def initialize_kalman():
    # Initialisierung des Kalman-Filters
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([0.])  # Anfangszustand
    kf.F = np.array([[1.]])  # Zustandsübergangsmatrix
    kf.H = np.array([[1.]])  # Beobachtungsmatrix
    kf.P *= 1000.  # Kovarianzmatrix
    kf.R = 5  # Beobachtungsrauschen
    kf.Q = 0.1  # Prozessrauschen
    return kf

def kalman_filter(series):
    # Kalman-Filter wird auf die Serie angewendet
    kf = initialize_kalman()
    filtered_data = []
    for measurement in series:
        kf.predict()
        kf.update([measurement])
        filtered_data.append(kf.x[0])
    return pd.Series(filtered_data)

def median_filter(series):
    # Median-Filter wird auf die Serie angewendet
    return series.rolling(window=3, center=True).median()

def okada_filter(series, alpha=10):
    # Okada-Filter wird auf die Serie angewendet
    filtered_series = series.copy()
    for t in range(1, len(series) - 1):
        x_t_minus_1 = series.iloc[t - 1]
        x_t = series.iloc[t]
        x_t_plus_1 = series.iloc[t + 1]
        filtered_series.iloc[t] = x_t + ((x_t_minus_1 + x_t_plus_1 - 2 * x_t) / (2 * (1 + np.exp(-alpha * (x_t - x_t_minus_1) * (x_t - x_t_plus_1)))))
    return filtered_series

def savitzky_golay_filter(series, window_size=5, polynom_order=2):
    # Savitzky-Golay-Filter wird auf die Serie angewendet
    return savgol_filter(series, window_size, polynom_order)

# Qualitätsmaße
def calculate_mse(df1, df2):
    # Mean Squared Error (MSE) wird berechnet
    mse_values = []
    for col in df1.columns:
        # NaN-Werte in beiden DataFrames werden ignoriert
        valid_idx = ~np.isnan(df1[col]) & ~np.isnan(df2[col])
        mse = mean_squared_error(df1[col][valid_idx], df2[col][valid_idx])
        mse_values.append(mse)
    return np.mean(mse_values)

def calculate_rmse(mse):
    # Root Mean Squared Error (RMSE) wird berechnet
    return np.sqrt(mse)

def calculate_wilcoxon(df1, df2):
    # Wilcoxon-Test wird berechnet
    wilcoxon_results = []
    for col in df1.columns:
        # Nur nicht-NA-Werte in beiden Reihen werden berücksichtigt
        valid_idx = df1[col].notna() & df2[col].notna()
        if valid_idx.any():  # Prüfen, ob nach Bereinigung noch Daten vorhanden sind
            stat, p_value = wilcoxon(df1[col][valid_idx], df2[col][valid_idx])
            wilcoxon_results.append(p_value)
        else:
            wilcoxon_results.append(np.nan)
    # Durchschnittlicher p-Wert wird berechnet, NaN-Werte werden ignoriert
    valid_p_values = [p for p in wilcoxon_results if not np.isnan(p)]
    if valid_p_values:  # Prüfen, ob gültige p-Werte vorhanden sind
        return np.nanmean(valid_p_values)
    else:
        return np.nan  # NaN zurückgeben, wenn keine gültigen p-Werte vorhanden sind

def calculate_correlation(df1, df2):
    # Korrelation zwischen zwei DataFrames wird berechnet
    correlations = [df1[col].corr(df2[col]) for col in df1.columns if not df1[col].isnull().all() and not df2[col].isnull().all()]
    return np.nanmean(correlations)

# Tkinter Dialoge
def get_file_path():
    # Ein Dialog zur Auswahl einer Datei wird geöffnet
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Wähle die Datei aus", filetypes=[("Tabellen", "*.xlsx *.xls *.csv")])
    root.destroy()
    return file_path

def save_filtered_data(df):
    # Ein Dialog zum Speichern der gefilterten Daten wird geöffnet
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(title="Speichere gefilterte Daten", filetypes=[("Excel files", "*.xlsx")])
    if file_path:
        df.to_excel(file_path, index=False)
        messagebox.showinfo("Erfolg", "Die Daten wurden erfolgreich gespeichert.")
    root.destroy()

def select_dataframe_to_save(filter_methods, df_original=None):
    def on_save():
        selected_method = combo.get()
        if selected_method == "Alle":
            save_all_filtered_dataframes(filter_methods, df_original)
        elif selected_method == "Original ohne Filter":
            if df_original is not None:
                save_filtered_dataframe(df_original, "Original_ohne_Filter")
            else:
                messagebox.showerror("Fehler", "Originaldaten sind nicht verfügbar.")
        else:
            save_filtered_dataframe(filter_methods[selected_method], selected_method)
        root.destroy()

    # Ein Dialog zur Auswahl des zu speichernden DataFrames wird geöffnet
    root = tk.Tk()
    root.title("Wähle den zu speichernden DataFrame")
    label = ttk.Label(root, text="Wähle den DataFrame:")
    label.pack(pady=10)
    combo_values = list(filter_methods.keys()) + ["Alle"]
    if df_original is not None:
        combo_values.append("Original ohne Filter")
    combo = ttk.Combobox(root, values=combo_values)
    combo.pack(pady=5)
    save_button = ttk.Button(root, text="Speichern", command=on_save)
    save_button.pack(pady=10)

    root.protocol("WM_DELETE_WINDOW", sys.exit)
    root.mainloop()

def save_filtered_dataframe(df, method_name):
    # Ein Dialog zum Speichern des gefilterten DataFrames wird geöffnet
    file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
    if file_path:
        df.to_excel(file_path, index=False)
        messagebox.showinfo("Information", f"{method_name} DataFrame gespeichert als {file_path}")

def save_all_filtered_dataframes(filter_methods, df_original):
    # Ein Dialog zur Auswahl eines Verzeichnisses zum Speichern aller gefilterten DataFrames wird geöffnet
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        zip_path = os.path.join(folder_selected, "gefilterte_Dataframes.zip")
        with ZipFile(zip_path, 'w') as zipf:
            if df_original is not None:
                temp_path = os.path.join(folder_selected, "Original_ohne_Filter.xlsx")
                df_original.to_excel(temp_path, index=False)
                zipf.write(temp_path, os.path.basename(temp_path))
                os.remove(temp_path)
            for method_name, df in filter_methods.items():
                # Temporäre Speicherung der .xlsx Datei
                temp_path = os.path.join(folder_selected, f"{method_name}.xlsx")
                df.to_excel(temp_path, index=False)
                zipf.write(temp_path, os.path.basename(temp_path))
                os.remove(temp_path)
        messagebox.showinfo("Information", f"Alle gefilterten DataFrames gespeichert als {zip_path}")

def save_statistics(results):
    # Ein Dialog zum Speichern der statistischen Werte wird geöffnet
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(title="Speichere statistische Werte", defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
    if file_path:
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.to_excel(file_path)
        messagebox.showinfo("Erfolg", "Die statistischen Werte wurden erfolgreich gespeichert.")
    root.destroy()

def ask_remove_background():
    # Ein Dialog zur Abfrage, ob die Hintergrundfluoreszenz entfernt werden soll, wird geöffnet
    root = tk.Tk()
    root.withdraw()
    remove_bg = messagebox.askyesno("Hintergrundfluoreszenz", "Soll die Hintergrundfluoreszenz entfernt werden?")
    root.destroy()
    return remove_bg

def main():
    file_path = get_file_path()
    if not file_path:
        print("Keine Datei ausgewählt.")
        return

    # Datei wird geladen, je nach Dateityp als CSV oder Excel
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, index_col=0)
    else:
        df = pd.read_excel(file_path, index_col=0)

    # Abfrage, ob die Hintergrundfluoreszenz entfernt werden soll
    remove_bg = ask_remove_background()
    if remove_bg:
        # Berechnung des Mittelwerts der letzten vier Spalten
        background_fluorescence = df.iloc[:, -4:].mean(axis=1)
        # Subtraktion des Mittelwerts von allen Spalten außer den letzten vier
        df_subtracted = df.iloc[:, :-4].sub(background_fluorescence, axis=0)
    else:
        # Behalte das ursprüngliche DataFrame ohne die letzten vier Spalten
        df_subtracted = df.iloc[:, :-4]

    df_subtracted.insert(0, 'Time', df.index)

    # Filtermethoden werden definiert
    filter_methods = {
        'Binomial': lambda series: binomial_filter(series),
        'Gaussian': lambda series: gaussian_smoothing(series, window_size=4, sigma=1.5),
        'Kalman': lambda series: kalman_filter(series),
        'Median': lambda series: median_filter(series),
        'Okada': lambda series: okada_filter(series, alpha=100),
        'Savitzky-Golay': lambda series: savitzky_golay_filter(series)
    }

    filter_methods_dataframes = {}
    results = {}

    # Filtermethoden werden auf die Daten angewendet und die Ergebnisse gespeichert
    for name, method in filter_methods.items():
        filtered_df = df_subtracted.drop('Time', axis=1).apply(method)
        filtered_df.insert(0, 'Time', df.index)
        filter_methods_dataframes[name] = filtered_df
        mse = calculate_mse(df_subtracted.drop('Time', axis=1), filtered_df.drop('Time', axis=1))
        rmse = calculate_rmse(mse)
        correlation = calculate_correlation(df_subtracted.drop('Time', axis=1), filtered_df.drop('Time', axis=1))
        w_stat = calculate_wilcoxon(df_subtracted.drop('Time', axis=1), filtered_df.drop('Time', axis=1))
        results[name] = {'MSE': mse, 'RMSE': rmse, 'Correlation': correlation, 'Wilcoxon': w_stat}

    # Statistische Werte werden gespeichert
    save_statistics(results)
    # Auswahl des zu speichernden DataFrames wird geöffnet
    select_dataframe_to_save(filter_methods_dataframes, df_subtracted if remove_bg else df)

    print("Prozess erfolgreich abgeschlossen.")
    sys.exit()

if __name__ == "__main__":
    main()
