import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import time
import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.signal import find_peaks, peak_widths

# Deconvolution einer Einzelneuronen-Zeitreihe
@njit(["float32[:], float32[:], float32[:], int64[:], float32[:], float32[:], float32, float32"], cache=True)
def oasis_trace(F, v, w, t, l, s, tau, fs):
    NT = F.shape[0]
    if np.isnan(tau):
        g = -1. / (2.0 * fs)  # Standardwert für g, wenn tau NaN ist
    else:
        g = -1. / (tau * fs)
    it = 0
    ip = 0
    while it < NT:
        v[ip], w[ip], t[ip], l[ip] = F[it], 1, it, 1
        while ip > 0:
            if v[ip - 1] * np.exp(g * l[ip - 1]) > v[ip]:
                f1 = np.exp(g * l[ip - 1])
                f2 = np.exp(2 * g * l[ip - 1])
                wnew = w[ip - 1] + w[ip] * f2
                v[ip - 1] = (v[ip - 1] * w[ip - 1] + v[ip] * w[ip] * f1) / wnew
                w[ip - 1] = wnew
                l[ip - 1] = l[ip - 1] + l[ip]
                ip -= 1
            else:
                break
        it += 1
        ip += 1
    s[t[1:ip]] = v[1:ip] - v[:ip - 1] * np.exp(g * l[:ip - 1])

# Deconvolution von Neuronen-Zeitreihen in Matrixform
@njit(["float32[:,:], float32[:,:], float32[:,:], int64[:,:], float32[:,:], float32[:,:], float32[:], float32"],
      parallel=True, cache=True)
def oasis_matrix(F, v, w, t, l, s, taus, fs):
    for n in prange(F.shape[0]):
        oasis_trace(F[n], v[n], w[n], t[n], l[n], s[n], taus[n], fs)

# Batchweise Deconvolution von Neuronen-Zeitreihen
def oasis(F: np.ndarray, batch_size: int, taus: np.ndarray, fs: float) -> np.ndarray:
    NN, NT = F.shape
    S = np.zeros((NN, NT), dtype=np.float32)
    for i in range(0, NN, batch_size):
        f = F[i:i + batch_size].astype(np.float32)
        v = np.zeros_like(f)
        w = np.zeros_like(f)
        t = np.zeros_like(f, dtype=np.int64)
        l = np.zeros_like(f)
        s = np.zeros_like(f)
        taus_batch = taus[i:i + batch_size].astype(np.float32)
        oasis_matrix(f, v, w, t, l, s, taus_batch, fs)
        S[i:i + batch_size] = s
    return S

# Berechnung von tau-Werten basierend auf Peaks in den Daten
def calculate_tau_from_peaks(data, fs, default_tau=2.0, min_distance=20):
    taus = []
    for i in range(data.shape[0]):
        signal = data[i, :]
        peaks, _ = find_peaks(signal, distance=min_distance)
        if len(peaks) > 1:
            results_half = peak_widths(signal, peaks, rel_height=0.5)
            tau = np.mean(results_half[0] / fs)
        else:
            tau = default_tau
        tau = tau * fs
        taus.append(tau)
    return np.array(taus, dtype=np.float32)

# Extraktion der Peak-Maxima aus einem Signal
def extract_peak_maxima(signal, min_distance):
    peaks, _ = find_peaks(signal, distance=min_distance)
    result = np.zeros_like(signal)
    result[peaks] = signal[peaks]
    return result

# Findet die Amplitude zwischen Startpunkt und lokalem Maximum
def find_amplitude(signal, peak):
    window_min = signal[max(0, peak - 5):peak + 1]
    local_min = min(window_min)
    local_min_idx = np.argmin(window_min) + max(0, peak - 5)

    window_max = signal[peak:min(peak + 25, len(signal))]
    local_max = max(window_max)
    local_max_idx = np.argmax(window_max) + peak

    amplitude = local_max - local_min
    return amplitude, local_min_idx, local_max_idx

# Berechnung der Frequenz der Amplituden
def calculate_frequencies(peaks, fs):
    intervals = np.diff(peaks) * 0.01  # Zeitpunkte zu Sekunden konvertieren
    if len(intervals) > 0:
        frequency = 1 / np.mean(intervals)
    else:
        frequency = 0
    return frequency

# Dialog zur Benutzereingabe von Parametern
def user_input_dialog():
    root = tk.Tk()
    root.title("Parameter Einstellungen")

    tk.Label(root, text="Sampling-Frequenz (Aufnahmefrequenz in Hz):").grid(row=0)
    tk.Label(root, text="Batch-Größe für Deconvolution (Anzahl der gleichzeitig zu verarbeitenden Neuronen):").grid(
        row=1)

    fs_entry = tk.Entry(root)
    batch_size_entry = tk.Entry(root)

    fs_entry.grid(row=0, column=1)
    batch_size_entry.grid(row=1, column=1)

    def on_submit():
        global fs_value, batch_size_value
        fs_value = float(fs_entry.get())
        batch_size_value = int(batch_size_entry.get())
        root.destroy()

    submit_button = tk.Button(root, text="Submit", command=on_submit)
    submit_button.grid(row=2, column=0, columnspan=2)
    root.mainloop()

    return fs_value, batch_size_value

# Hauptfunktion
def main():
    # Benutzerdefinierte Eingabe von Sampling-Frequenz und Batch-Größe
    fs, batch_size = user_input_dialog()

    # Öffnen eines Dialogs zur Auswahl der Excel-Datei
    file_path = filedialog.askopenfilename(title="Excel-Datei auswählen", filetypes=[("Excel files", "*.xlsx *.xls")])
    if not file_path:
        messagebox.showerror("Fehler", "Keine Datei ausgewählt.")
        return

    # Laden der Excel-Datei
    df = pd.read_excel(file_path)
    start_time = time.time()
    F = df.iloc[:, 1:].values.T  # Ignoriere die Zeit-Spalte

    # Berechnung von tau-Werten für jede Zeile in den Daten
    taus = calculate_tau_from_peaks(F, fs, default_tau=2.0, min_distance=20)

    # Durchführung der Deconvolution mit den berechneten tau-Werten
    S = oasis(F, batch_size, taus, fs)

    # Extraktion der Peak-Maxima aus den dekonvolvierten Signalen
    S_peaks = np.array([extract_peak_maxima(s, min_distance=20) for s in S])

    amplitudes = []
    frequencies = []
    min_indices = []
    max_indices = []
    for i in range(S.shape[0]):
        peaks, _ = find_peaks(S_peaks[i, :], distance=20)
        amp_min_max = [find_amplitude(F[i, :], peak) for peak in peaks]
        amp, min_idx, max_idx = zip(*amp_min_max)
        amplitudes.append(amp)
        frequencies.append(calculate_frequencies(peaks, fs))
        min_indices.append(min_idx)
        max_indices.append(max_idx)

    columns = df.columns[1:]

    # Erstellen von DataFrames für die Ergebnisse
    denoised_df = pd.DataFrame(F.T, columns=columns)
    deconvolved_df = pd.DataFrame(S_peaks.T, columns=columns)
    taus_df = pd.DataFrame(taus, columns=["tau"])
    amplitudes_df = pd.DataFrame(amplitudes).T
    frequencies_df = pd.DataFrame(frequencies, columns=["frequency"])

    end_time = time.time()
    print(f"CPU time: {end_time - start_time} seconds")

    # Öffnen eines Dialogs zum Speichern der Ergebnisse
    save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", title="Speicherort wählen",
                                             filetypes=[("Excel files", " .xlsx *.xls")])
    if save_path:
        # Speichern der Ergebnisse in einer Excel-Datei mit mehreren Arbeitsblättern
        with pd.ExcelWriter(save_path) as writer:
            denoised_df.to_excel(writer, sheet_name='denoised')
            deconvolved_df.to_excel(writer, sheet_name='deconvolved')
            taus_df.to_excel(writer, sheet_name='tau')
            amplitudes_df.to_excel(writer, sheet_name='amplitude')
            frequencies_df.to_excel(writer, sheet_name='frequency')
        messagebox.showinfo("Fertig", "Daten erfolgreich verarbeitet und gespeichert.")

# Start des Hauptprogramms
if __name__ == "__main__":
    main()

