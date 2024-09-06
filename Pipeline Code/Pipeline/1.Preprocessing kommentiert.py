from skimage import exposure, io, filters, img_as_float
from skimage.exposure import adjust_gamma
from skimage.filters import unsharp_mask, median
import matplotlib.pyplot as plt
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog

# Tkinter-Initialisierung: Ein unsichtbares Hauptfenster wird erstellt, um den Datei-Explorer zu verwenden
root = tk.Tk()
root.withdraw()

# Ein Datei-Explorer Dialog wird geöffnet und nach dem Verzeichnis gefragt
directory_path = filedialog.askdirectory()

# Es wird überprüft, ob ein Verzeichnis ausgewählt wurde
if directory_path:
    print(f"Das ausgewählte Verzeichnis ist: {directory_path}")
else:
    print("Es wurde kein Verzeichnis ausgewählt.")

# Eine Liste aller Dateinamen im Verzeichnis wird erstellt
all_files = os.listdir(directory_path)

# Nur die TIFF-Dateien werden herausgefiltert und sortiert, falls nötig
tiff_files = [f for f in all_files if f.endswith('.tif')]
tiff_files.sort()  # Sortieren, wenn die Reihenfolge wichtig ist

# Die ersten 20 TIFF-Bilder werden geladen und die Farbkanäle getrennt
green_channel_images = []
for file_name in tiff_files[:100]:
    image_path = os.path.join(directory_path, file_name)
    image = io.imread(image_path)

    # Der grüne Farbkanal (Index 1) wird extrahiert
    green_channel = image[:, :, 1]
    green_channel_images.append(green_channel)

# Eine Z-Projektion wird durchgeführt, um die grünen Kanäle zu einem Bild zu fusionieren
# Der Mittelwert über alle Bilder wird genommen
green_projection = np.mean(green_channel_images, axis=0)

# Eine Rauschreduktion wird mittels Median-Filter durchgeführt
image_denoised = median(green_projection, behavior='ndimage')
# Das Bild wird zu Fließkommazahlenformat konvertiert und die Intensitäten werden reskaliert
image_denoised = img_as_float(image_denoised)
image_denoised = exposure.rescale_intensity(image_denoised, in_range='image', out_range=(0, 1))

# Eine benutzerdefinierte Dialogklasse für die Eingabe der Bildverarbeitungsparameter wird erstellt
class CustomDialog(simpledialog.Dialog):
    def __init__(self, parent, title=None):
        self.clip_limit = tk.DoubleVar()
        self.radius = tk.DoubleVar()
        self.amount = tk.DoubleVar()
        self.gamma_value = tk.DoubleVar()  # Hinzugefügte Variable für den Gamma-Wert
        super().__init__(parent, title)

    def body(self, master):
        # Eingabefelder für die verschiedenen Parameter werden erstellt
        tk.Label(master, text="CLAHE Clip-Limit (0.01):").grid(row=0)
        tk.Label(master, text="Unsharp Mask Radius (2.0):").grid(row=1)
        tk.Label(master, text="Unsharp Mask Amount (1.0):").grid(row=2)
        tk.Label(master, text="Gamma-Wert (2.0):").grid(row=3)

        tk.Entry(master, textvariable=self.clip_limit).grid(row=0, column=1)
        tk.Entry(master, textvariable=self.radius).grid(row=1, column=1)
        tk.Entry(master, textvariable=self.amount).grid(row=2, column=1)
        tk.Entry(master, textvariable=self.gamma_value).grid(row=3, column=1)
        return master  # initial focus

    def apply(self):
        # Die Eingabewerte werden in der result-Variable gespeichert
        self.result = (self.clip_limit.get(), self.radius.get(), self.amount.get(), self.gamma_value.get())

# Ein benutzerdefinierter Dialog für Bildverarbeitungsparameter wird erstellt und angezeigt
dialog = CustomDialog(root, title="Parameter für Bildverarbeitung (Stabile Parameter in Klammern)")

if dialog.result:
    clip_limit, radius, amount, gamma_value = dialog.result

    # CLAHE (Kontrastlimitierte adaptive Histogramm-Equalisierung) wird auf das rauschreduzierte Bild angewendet
    image_clahe = exposure.equalize_adapthist(image_denoised, clip_limit=clip_limit)

    # Eine Kantenverstärkung wird mittels Unsharp Mask durchgeführt
    image_sharpened = unsharp_mask(image_clahe, radius=radius, amount=amount)

    # Eine Gamma-Korrektur wird angewendet
    gamma_corrected = adjust_gamma(image_sharpened, gamma=gamma_value)

    # Die Ergebnisse werden angezeigt
    fig, ax = plt.subplots(1, 4, figsize=(20, 10))

    ax[0].imshow(green_projection, cmap='gray')
    ax[0].set_title('Z-Projection')
    ax[0].axis('off')

    ax[1].imshow(image_denoised, cmap='gray')
    ax[1].set_title('Median-Filter')
    ax[1].axis('off')

    ax[2].imshow(image_clahe, cmap='gray')
    ax[2].set_title('CLAHE')
    ax[2].axis('off')

    ax[3].imshow(gamma_corrected, cmap='gray')
    ax[3].set_title('Gamma-Korrektur')
    ax[3].axis('off')

    plt.tight_layout()
    plt.show()

    # Das Bild wird gespeichert
    output_path = filedialog.asksaveasfilename(defaultextension=".tif",
                                               filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")])
    if output_path:
        io.imsave(output_path, gamma_corrected)
        print("Das Bild wurde erfolgreich gespeichert unter:", output_path)
    else:
        print("Das Speichern wurde abgebrochen.")
else:
    print("Keine Werte eingegeben.")
