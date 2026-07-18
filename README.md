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

## Projectstructuur

```text
LabFit/
├── app.py
├── requirements.txt
├── render.yaml
├── sample_data.csv
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Publiceren via Render

1. Upload de volledige map naar een GitHub-repository.
2. Maak in Render een nieuwe Web Service aan.
3. Koppel de repository.
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn app:app`

Of laat Render automatisch de instellingen uit `render.yaml` gebruiken.

## Belangrijke wetenschappelijke opmerking

R² en RMSE zijn hulpmiddelen, geen bewijs dat een model fysisch correct is. Inspecteer ook de residuen, parameteronzekerheden en aannames van het model.
