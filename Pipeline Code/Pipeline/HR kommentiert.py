import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageSequence
import numpy as np
import os, tifffile, time
from threading import Thread
import psutil

# Funktion zur Auswahl eines Verzeichnisses über einen Dialog
def ask_directory():
    root = tk.Tk()
    root.withdraw()  # Das Tkinter-Hauptfenster wird verborgen
    folder_path = filedialog.askdirectory(title="Wähle den Ordner mit den TIFF-Dateien")
    root.destroy()
    return folder_path

# Funktion zum Laden und Extrahieren des grünen Kanals aus den Bildern
def load_and_extract_green_channel(folder_path, progress_callback, max_memory_usage):
    # Liste aller TIFF-Dateien im ausgewählten Verzeichnis
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tif') or f.endswith('.tiff')]
    image_files.sort()  # Dateien werden in alphabetischer Reihenfolge sortiert

    green_images = []  # Liste zur Speicherung der grünen Kanäle
    total_files = len(image_files)
    for idx, image_file in enumerate(image_files):
        img = Image.open(image_file)
        for frame in ImageSequence.Iterator(img):
            green_channel = np.array(frame)[:, :, 1]  # Extrahieren des grünen Kanals
            green_images.append(green_channel)
        progress_callback(idx + 1, total_files)
        # Überwachung der Speicherauslastung
        current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        max_memory_usage[0] = max(max_memory_usage[0], current_memory)

    return np.stack(green_images)  # Rückgabe der gestapelten grünen Kanäle

# Funktion zum Speichern des Bildstapels als TIFF-Datei
def save_image_stack_tifffile(image_stack, save_path=None):
    if save_path is None:
        root = tk.Tk()
        root.withdraw()  # Das Tkinter-Hauptfenster wird verborgen
        save_path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIFF files", "*.tif")], title="Speichere den Bilderstack")
        root.destroy()

    if save_path:  # Sicherstellen, dass ein Speicherpfad angegeben wurde
        tifffile.imwrite(save_path, image_stack, bigtiff=True)

# Funktion zur Aktualisierung der Fortschrittsanzeige
def update_progressbar(progress, total, progress_var):
    progress_var.set((progress / total) * 100)

# Funktion zum Starten des Lade- und Speichervorgangs
def start_processing(folder_path, progress_var, root, label, max_memory_usage):
    image_stack = load_and_extract_green_channel(folder_path, lambda p, t: update_progressbar(p, t, progress_var), max_memory_usage)
    save_image_stack_tifffile(image_stack)
    label.config(text="Laden abgeschlossen")  # Text wird aktualisiert, um den Abschluss anzuzeigen
    root.after(2000, root.destroy)  # Fenster wird nach 2 Sekunden geschlossen

# Hauptfunktion
def main():
    start_time = time.time()  # Startzeit
    max_memory_usage = [0]  # Speicher für maximale RAM-Auslastung

    root = tk.Tk()
    root.title("Bildersequenz Laden und Speichern")

    label = tk.Label(root, text="Laden der Bildersequenz")
    label.pack(pady=10)

    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
    progress_bar.pack(expand=True, fill=tk.X, padx=40, pady=10)

    folder_path = ask_directory()
    if folder_path:
        # Thread wird gestartet, um das Laden und Speichern der Bilder im Hintergrund durchzuführen
        thread = Thread(target=start_processing, args=(folder_path, progress_var, root, label, max_memory_usage))
        thread.start()
    else:
        root.destroy()  # Fenster wird geschlossen, wenn kein Ordner ausgewählt wurde

    root.mainloop()

    end_time = time.time()  # Endzeit
    print(f"Total Runtime: {end_time - start_time:.2f} seconds")
    print(f"Max RAM usage: {max_memory_usage[0]:.2f} MB")

if __name__ == "__main__":
    main()
