import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox, Toplevel, Label, Entry, Button, StringVar, IntVar, ttk
import tkinter as tk

# Definition einer exponentiellen Abklingfunktion
def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

# Definition einer Gauss'schen Funktion
def gaussian(x, mean, std):
    return norm.pdf(x, mean, std)

# Berechnung der vollen Breite bei halbem Maximum (FWHM) für eine Normalverteilung
def fwhm(sigma):
    return 2.3548 * sigma

# Anpassung einer Gauss'schen Kurve an ein Histogramm und Rückgabe der Parameter
def fit_gaussian(hist, bins):
    mid_bin = (bins[:-1] + bins[1:]) / 2
    initial_guess = [np.mean(mid_bin), np.std(mid_bin)]
    params, cov = curve_fit(gaussian, mid_bin, hist, p0=initial_guess)
    return params

# Berechnung des Delta-F-over-F (ΔF/F) Werts für eine gegebene Zeitreihe
def calculate_dff(series):
    x = np.arange(len(series))
    params, _ = curve_fit(exp_decay, x, series, bounds=(0, [np.inf, np.inf, np.inf]), maxfev=10000)
    fitted_curve = exp_decay(x, *params)
    errors = series - fitted_curve
    hist, bins = np.histogram(errors, bins='auto', density=True)
    gauss_params = fit_gaussian(hist, bins)
    noise_range = fwhm(gauss_params[1])
    noise_values = errors[np.abs(errors) <= noise_range / 2]
    if len(noise_values) > 0:
        params, _ = curve_fit(exp_decay, x, series + noise_values.mean(), p0=params, bounds=(0, [np.inf, np.inf, np.inf]), maxfev=10000)
    else:
        params, _ = curve_fit(exp_decay, x, series, p0=params, bounds=(0, [np.inf, np.inf, np.inf]), maxfev=10000)
    baseline = exp_decay(x, *params)
    dff = ((series - baseline) / baseline) * 100
    return dff, baseline

# Normalisierung einer Zeitreihe, indem der kleinste Wert auf 0 gesetzt wird
def normalize_series(series):
    min_value = np.min(series)
    return series - min_value

# Hauptfunktion
def main():
    root = tk.Tk()
    root.withdraw()

    # Öffnen eines Dialogs zur Dateiauswahl
    file_path = filedialog.askopenfilename(title="Wähle die Datei aus", filetypes=[("Tabellen", "*.xlsx *.xls *.csv")])
    if not file_path:
        messagebox.showerror("Fehler", "Keine Datei ausgewählt.")
        return

    # Laden der Daten aus der ausgewählten Datei
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, index_col=0)
    else:
        df = pd.read_excel(file_path, index_col=0)

    root.deiconify()
    # Erstellen einer Fortschrittsanzeige
    progress_bar = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=300, mode='determinate')
    progress_bar.pack()
    root.update()

    # Berechnung von ΔF/F für jede Spalte in der Tabelle und Aktualisierung der Fortschrittsanzeige
    for i, col in enumerate(df.columns):
        series = df[col].values
        dff, _ = calculate_dff(series)
        normalized_dff = normalize_series(dff)  # Normalisieren, damit der kleinste Wert 0 ist
        df[col] = normalized_dff
        progress_bar['value'] = (i + 1) * 100 / len(df.columns)
        root.update()

    progress_bar.destroy()

    # Öffnen eines Dialogs zum Speichern der Ergebnisse
    save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx *.xls")])
    if save_path:
        df.to_excel(save_path)
        messagebox.showinfo("Fertig", f"ΔF/F Daten wurden in '{save_path}' gespeichert.")

    root.destroy()

# Start des Hauptprogramms
if __name__ == "__main__":
    main()
