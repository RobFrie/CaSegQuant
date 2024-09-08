import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

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

def find_peak_and_start(signal, fs, min_start_time):
    peak_index = np.argmax(signal[150:]) + 150
    peak_value = signal[peak_index]

    # Calculate the baseline as the average signal value before the search period
    baseline_value = np.mean(signal[:min_start_time])

    amplitude = peak_value - baseline_value

    # Find the start of the peak
    start_index = min_start_time
    for i in range(peak_index - 1, min_start_time - 1, -1):
        if signal[peak_index] - signal[i] >= 5:
            start_index = i
            break

    # Calculate decay time (tau) starting from the peak
    decay_time_indices = np.arange(peak_index, len(signal))
    decay_time_values = signal[peak_index:]

    if np.any(decay_time_values <= peak_value * (1 / np.e)):
        decay_end_index = np.argmax(decay_time_values <= peak_value * (1 / np.e))
        tau = (decay_time_indices[decay_end_index] - decay_time_indices[0]) / fs
        tau_end_index = decay_time_indices[decay_end_index]
    else:
        tau = calculate_tau_with_trend(decay_time_values, decay_time_indices, peak_value, fs)
        tau_end_index = peak_index + int(tau * fs)

    return peak_index, start_index, amplitude, tau, tau_end_index

def get_thresholds():
    root = tk.Tk()
    root.title("Input Thresholds")

    tk.Label(root, text="Sampling Rate (fs):").grid(row=0, column=0)
    tk.Label(root, text="Low Tau Threshold:").grid(row=1, column=0)
    tk.Label(root, text="Late Peak Threshold (time points):").grid(row=2, column=0)
    tk.Label(root, text="Min Start Time (time points):").grid(row=3, column=0)

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

def plot_signal_with_annotations(series, fs, min_start_time, late_peak_threshold, peak_index, tau, tau_end_index, col_name):
    plt.figure(figsize=(12, 6))
    time_axis = np.arange(len(series)) / fs
    plt.plot(time_axis, series, label='Signal')

    # Shading min start time area
    plt.axvspan(0, min_start_time / fs, color='purple', alpha=0.3, label='Baseline')
    # Shading late peak threshold area
    plt.axvspan((len(series) - late_peak_threshold) / fs, len(series) / fs, color='orange', alpha=0.3, label='Late Peak Threshold')
    # Marking peak maximum with a cross
    plt.scatter([peak_index / fs], [series[peak_index]], color='red', marker='x', label='Peak Maximum', zorder=5)
    # Shading tau area
    if not np.isnan(tau):
        plt.axvspan(peak_index / fs, tau_end_index / fs, color='blue', alpha=0.3, label='Tau Area (37%)')

    # Annotating tau value
    if not np.isnan(tau):
        plt.text(peak_index / fs, series[peak_index], f'Tau: {tau:.2f}', horizontalalignment='right')

    plt.title(f'Signal Analysis: {col_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

def main():
    fs, low_tau_threshold, late_peak_threshold, min_start_time = get_thresholds()

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(title="Excel-Datei auswählen", filetypes=[("Excel files", "*.xlsx *.xls")])
    if not file_path:
        messagebox.showerror("Fehler", "Keine Datei ausgewählt.")
        return

    df = pd.read_excel(file_path)
    time_series = df.iloc[:, 1:].values.T  # Ignore the time column
    column_names = df.columns[1:]  # Capture the original column names

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

    for col_name, series in zip(column_names, time_series):
        peak_index, start_index, amplitude, tau, tau_end_index = find_peak_and_start(series, fs, min_start_time)
        time_to_max = peak_index - start_index
        end_time_index = len(series) - 1

        plot_signal_with_annotations(series, fs, min_start_time, late_peak_threshold, peak_index, tau, tau_end_index, col_name)

        if peak_index > end_time_index - late_peak_threshold:
            late_peaks_results["Column Name"].append(col_name)
            late_peaks_results["Peak Time"].append(peak_index)
            late_peaks_results["Start Time"].append(start_index)
            late_peaks_results["Time to Max"].append(time_to_max)
            late_peaks_results["Amplitude"].append(amplitude)
            late_peaks_results["Tau"].append(tau if not np.isnan(tau) else "NaN")
        elif np.isnan(tau):
            unrealistic_tau_results["Column Name"].append(col_name)
            unrealistic_tau_results["Peak Time"].append(peak_index)
            unrealistic_tau_results["Start Time"].append(start_index)
            unrealistic_tau_results["Time to Max"].append(time_to_max)
            unrealistic_tau_results["Amplitude"].append(amplitude)
            unrealistic_tau_results["Tau"].append("NaN")
        elif tau < low_tau_threshold:
            low_tau_results["Column Name"].append(col_name)
            low_tau_results["Peak Time"].append(peak_index)
            low_tau_results["Start Time"].append(start_index)
            low_tau_results["Time to Max"].append(time_to_max)
            low_tau_results["Amplitude"].append(amplitude)
            low_tau_results["Tau"].append(tau)
        else:
            results["Column Name"].append(col_name)
            results["Peak Time"].append(peak_index)
            results["Start Time"].append(start_index)
            results["Time to Max"].append(time_to_max)
            results["Amplitude"].append(amplitude)
            results["Tau"].append(tau)

    results_df = pd.DataFrame(results)
    unrealistic_tau_df = pd.DataFrame(unrealistic_tau_results)
    late_peaks_df = pd.DataFrame(late_peaks_results)
    low_tau_df = pd.DataFrame(low_tau_results)

    save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", title="Speicherort wählen", filetypes=[("Excel files", "*.xlsx *.xls")])
    if save_path:
        with pd.ExcelWriter(save_path) as writer:
            results_df.to_excel(writer, sheet_name='Results', index=False)
            unrealistic_tau_df.to_excel(writer, sheet_name='Unrealistic Tau', index=False)
            late_peaks_df.to_excel(writer, sheet_name='Late Peaks', index=False)
            low_tau_df.to_excel(writer, sheet_name='Low Tau', index=False)
        messagebox.showinfo("Fertig", "Daten erfolgreich verarbeitet und gespeichert.")

    root.destroy()

if __name__ == "__main__":
    main()

