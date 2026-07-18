# LabFit

Een compacte Flask-webapp voor natuurkundige data-analyse.

## Functies

- CSV uploaden of meetdata plakken
- Kolommen `x`, `y`, optioneel `sigma_x` en `sigma_y`
- Lineaire, exponentiële en power-law fits
- Orthogonal Distance Regression, zodat volledige x- en y-onzekerheden gebruikt kunnen worden
- Parameteronzekerheden, R² en RMSE
- PNG-, CSV- en JSON-downloads
- Kopieerbare rapporttekst

## Lokaal starten op Windows

Open PowerShell in de projectmap en voer uit:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py app.py
```

Open daarna `http://127.0.0.1:5000`.

