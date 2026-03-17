# AdPro Assignment
Project Okavango is a lightweight environmental data analysis tool built with Python, GeoPandas, and Streamlit. It uses the most recent datasets from Our World in Data to analyze deforestation, forest change, protected land, and land degradation, integrating geospatial country data to deliver interactive global visualizations.

## Group N
This project was developed by:

- Margarida Rodrigues, student number 71712, 71712@novasbe.pt
- Joao Roque, student number 73047, 73047@novasbe.pt
- Nicolas Oteri, student number 71642, 71642@novasbe.pt
- Karl Harfouche, student number 70044, 70044@novasbe.pt

## Streamlit app
The app now includes:

- An environmental dashboard with world maps, yearly filtering, and country-level comparisons.
- An AI workflow page that downloads ESRI World Imagery for chosen coordinates, stores the image in `images/`, generates a local vision-model description through Ollama, and runs a second Ollama model to flag environmental risk.

Run it with:

```powershell
py -m streamlit run app\streamlit_app.py
```

