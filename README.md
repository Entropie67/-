# Visualiseur du problème à trois corps

Petite app Python (Streamlit) pour explorer la dynamique chaotique du problème à trois corps, visualiser des trajectoires 3D et jouer avec des paramètres depuis un panneau latéral.

## Lancer en local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

L'app propose:
- des presets (dont une figure en 8),
- la modification des masses,
- la durée de simulation et la résolution,
- un mode libre pour changer les conditions initiales corps par corps.
