import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

# Definition einer exponentiellen Abklingfunktion
def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

# Berechnung von tau unter Berücksichtigung eines Trends
def calculate_tau_with_trend(y, x, peak_value, fs):
    target_value = peak_value * (1 / np.e)
    linear_reg = LinearRegression()
    linear_reg.fit(x.reshape(-1, 1), y)
    trend = linear_reg.coef_[0]
    intercept = linear_reg.intercept_

    if trend == 0:
        return np.nan

    time_to_target = (target_value - intercept) / trend
    if time_to_target < 0:
        return np.nan
    tau = time_to_target / fs
    return tau

# Finden des Peaks und Startpunktes im Signal
def find_peak_and_start(signal, fs, min_start_time):
    peak_index = np.argmax(signal[150:]) + 150
    peak_value = signal[peak_index]

    # Berechnung des Basiswerts als Durchschnitt vor der Suchperiode
    baseline_value = np.mean(signal[:min_start_time])

    amplitude = peak_value - baseline_value

    # Finden des Startpunkts des Peaks
    start_index = min_start_time
    for i in range(peak_index - 1, min_start_time - 1, -1):
        if signal[peak_index] - signal[i] >= 5:
            start_index = i
            break

    # Berechnung der Abklingzeit (tau) beginnend vom Peak
    decay_time_indices = np.arange(peak_index, len(signal))
    decay_time_values = signal[peak_index:]

    if np.any(decay_time_values <= peak_value * (1 / np.e)):
        decay_end_index = np.argmax(decay_time_values <= peak_value * (1 / np.e))
        tau = (decay_time_indices[decay_end_index] - decay_time_indices[0]) / fs
    else:
        tau = calculate_tau_with_trend(decay_time_values, decay_time_indices, peak_value, fs)

    return peak_index, start_index, amplitude, tau

# Dialog zur Benutzereingabe von Parametern
def get_thresholds():
    root = tk.Tk()
    root.title("Input Thresholds")

    tk.Label(root, text="Sampling Rate (Aufnahmefrequenz in Hz):").grid(row=0, column=0)
    tk.Label(root, text="Low Tau Threshold (in Sekunden):").grid(row=1, column=0)
    tk.Label(root, text="Late Peak Threshold (Zeitpunkte || 10 = 1 Sekunde):").grid(row=2, column=0)
    tk.Label(root, text="Min Start Time (Zeitpunkte || 10 = 1 Sekunde):").grid(row=3, column=0)

    fs_entry = tk.Entry(root)
    low_tau_entry = tk.Entry(root)
    late_peak_entry = tk.Entry(root)
    min_start_time_entry = tk.Entry(root)

    fs_entry.grid(row=0, column=1)
    low_tau_entry.grid(row=1, column=1)
    late_peak_entry.grid(row=2, column=1)
    min_start_time_entry.grid(row=3, column=1)

    def on_submit():
        global fs_value, low_tau_threshold, late_peak_threshold, min_start_time
        fs_value = float(fs_entry.get())
        low_tau_threshold = float(low_tau_entry.get())
        late_peak_threshold = int(late_peak_entry.get())
        min_start_time = int(min_start_time_entry.get())
        root.destroy()

    submit_button = tk.Button(root, text="Submit", command=on_submit)
    submit_button.grid(row=4, column=0, columnspan=2)

    root.mainloop()

    return fs_value, low_tau_threshold, late_peak_threshold, min_start_time

# Hauptfunktion
def main():
    # Abruf der Parameter vom Benutzer
    fs, low_tau_threshold, late_peak_threshold, min_start_time = get_thresholds()

    root = tk.Tk()
    root.withdraw()

    # Öffnen eines Dialogs zur Auswahl der Excel-Datei
    file_path = filedialog.askopenfilename(title="Excel-Datei auswählen", filetypes=[("Excel files", "*.xlsx *.xls")])
    if not file_path:
        messagebox.showerror("Fehler", "Keine Datei ausgewählt.")
        return

    # Laden der Excel-Datei
    df = pd.read_excel(file_path)
    time_series = df.iloc[:, 1:].values.T  # Ignoriere die Zeit-Spalte
    column_names = df.columns[1:]  # Erfassen der ursprünglichen Spaltennamen

    # Initialisieren der Ergebnis-DataFrames
    results = {
        "Column Name": [],
        "Peak Time": [],
        "Start Time": [],
        "Time to Max": [],
        "Amplitude": [],
        "Tau": []
    }

    unrealistic_tau_results = {
        "Column Name": [],
        "Peak Time": [],
        "Start Time": [],
        "Time to Max": [],
        "Amplitude": [],
        "Tau": []
    }

    late_peaks_results = {
        "Column Name": [],
        "Peak Time": [],
        "Start Time": [],
        "Time to Max": [],
        "Amplitude": [],
        "Tau": []
    }

    low_tau_results = {
        "Column Name": [],
        "Peak Time": [],
        "Start Time": [],
        "Time to Max": [],
        "Amplitude": [],
        "Tau": []
    }

    # Verarbeitung jeder Zeile der Zeitreihe
    for col_name, series in zip(column_names, time_series):
        peak_index, start_index, amplitude, tau = find_peak_and_start(series, fs, min_start_time)
        time_to_max = peak_index - start_index
        end_time_index = len(series) - 1
        if peak_index > end_time_index - late_peak_threshold:
            # Spät auftretende Peaks
            late_peaks_results["Column Name"].append(col_name)
            late_peaks_results["Peak Time"].append(peak_index)
            late_peaks_results["Start Time"].append(start_index)
            late_peaks_results["Time to Max"].append(time_to_max)
            late_peaks_results["Amplitude"].append(amplitude)
            late_peaks_results["Tau"].append(tau if not np.isnan(tau) else "NaN")
        elif np.isnan(tau):
            # Unrealistische Tau-Werte
            unrealistic_tau_results["Column Name"].append(col_name)
            unrealistic_tau_results["Peak Time"].append(peak_index)
            unrealistic_tau_results["Start Time"].append(start_index)
            unrealistic_tau_results["Time to Max"].append(time_to_max)
            unrealistic_tau_results["Amplitude"].append(amplitude)
            unrealistic_tau_results["Tau"].append("NaN")
        elif tau < low_tau_threshold:
            # Niedrige Tau-Werte
            low_tau_results["Column Name"].append(col_name)
            low_tau_results["Peak Time"].append(peak_index)
            low_tau_results["Start Time"].append(start_index)
            low_tau_results["Time to Max"].append(time_to_max)
            low_tau_results["Amplitude"].append(amplitude)
            low_tau_results["Tau"].append(tau)
        else:
            # Normale Ergebnisse
            results["Column Name"].append(col_name)
            results["Peak Time"].append(peak_index)
            results["Start Time"].append(start_index)
            results["Time to Max"].append(time_to_max)
            results["Amplitude"].append(amplitude)
            results["Tau"].append(tau)

    # Konvertierung der Ergebnisse in DataFrames
    results_df = pd.DataFrame(results)
    unrealistic_tau_df = pd.DataFrame(unrealistic_tau_results)
    late_peaks_df = pd.DataFrame(late_peaks_results)
    low_tau_df = pd.DataFrame(low_tau_results)

    # Öffnen eines Dialogs zum Speichern der Ergebnisse
    save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", title="Speicherort wählen", filetypes=[("Excel files", "*.xlsx *.xls")])
    if save_path:
        with pd.ExcelWriter(save_path) as writer:
            results_df.to_excel(writer, sheet_name='Results', index=False)
            unrealistic_tau_df.to_excel(writer, sheet_name='Unrealistic Tau', index=False)
            late_peaks_df.to_excel(writer, sheet_name='Late Peaks', index=False)
            low_tau_df.to_excel(writer, sheet_name='Low Tau', index=False)
        messagebox.showinfo("Fertig", "Daten erfolgreich verarbeitet und gespeichert.")

    root.destroy()

# Start des Hauptprogramms
if __name__ == "__main__":
    main()
