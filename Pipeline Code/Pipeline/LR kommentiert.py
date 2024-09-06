import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageSequence
import numpy as np
import os, tifffile, sys, time
from threading import Thread
import psutil

# Funktion zur Auswahl eines Verzeichnisses über einen Dialog
def ask_directory():
    root = tk.Tk()
    root.withdraw()  # Das Tkinter-Hauptfenster wird verborgen
    folder_path = filedialog.askdirectory(title="Wähle den Ordner mit den TIFF-Dateien")
    root.destroy()
    return folder_path

# Funktion zur Auswahl eines Speicherpfades über einen Dialog
def ask_save_path():
    root = tk.Tk()
    root.withdraw()  # Das Tkinter-Hauptfenster wird verborgen
    save_path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIFF files", "*.tif")], title="Speichere den Bilderstack")
    root.destroy()
    return save_path

# Funktion zum Verarbeiten und Speichern der Bilder
def process_and_save_images(folder_path, save_path, progress_callback, max_memory_usage):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tif') or f.endswith('.tiff')]
    image_files.sort()
    total_files = len(image_files)
    with tifffile.TiffWriter(save_path, bigtiff=True) as tif:
        for idx, image_file in enumerate(image_files):
            img = Image.open(image_file)
            for frame in ImageSequence.Iterator(img):
                green_channel = np.array(frame)[:, :, 1]
                tif.write(green_channel)
            progress_callback(idx + 1, total_files)
            # Überwachung der Speicherauslastung
            current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            max_memory_usage[0] = max(max_memory_usage[0], current_memory)

# Funktion zur Aktualisierung der Fortschrittsanzeige
def update_progressbar(progress, total, progress_var, root):
    root.update_idletasks()
    progress_var.set((progress / total) * 100)

# Funktion zum Speichern der Bilder
def save_images(folder_path, progress_var, root, label, max_memory_usage):
    save_path = ask_save_path()
    if save_path:
        process_and_save_images(folder_path, save_path, lambda p, t: update_progressbar(p, t, progress_var, root), max_memory_usage)
        label.config(text="Laden abgeschlossen")
        root.after(1000, root.destroy)
        root.after(1000, sys.exit)
    else:
        root.destroy()

# Hauptfunktion
def main():
    start_time = time.time()  # Startzeit
    max_memory_usage = [0]  # Initialisierung der Liste für die maximale Speicherauslastung

    root = tk.Tk()
    root.title("Bildersequenz Laden und Speichern")

    label = tk.Label(root, text="Laden der Bildersequenz")
    label.pack(pady=10)

    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
    progress_bar.pack(expand=True, fill=tk.X, padx=40, pady=10)

    folder_path = ask_directory()
    if folder_path:
        # Thread wird gestartet, um das Speichern der Bilder im Hintergrund durchzuführen
        thread = Thread(target=save_images, args=(folder_path, progress_var, root, label, max_memory_usage))
        thread.start()
    else:
        root.destroy()  # Fenster wird geschlossen, wenn kein Ordner ausgewählt wurde

    root.mainloop()

    end_time = time.time()  # Endzeit
    print(f"Total Runtime: {end_time - start_time:.2f} seconds")
    print(f"Max RAM usage: {max_memory_usage[0]:.2f} MB")

if __name__ == "__main__":
    main()