# Project Okavango - Part 2

Project Okavango is a lightweight environmental monitoring prototype built for the Advanced Programming assignment. It combines Our World in Data datasets, geospatial country maps, ESRI World Imagery, and local Ollama models to help users explore environmental patterns and run a first-pass AI-based risk screening workflow.

## Group N

This project was developed by:

- Margarida Rodrigues, student number 71712, 71712@novasbe.pt
- Joao Roque, student number 73047, 73047@novasbe.pt
- Nicolas Oteri, student number 71642, 71642@novasbe.pt
- Karl Harfouche, student number 70044, 70044@novasbe.pt

## What The App Does

The Streamlit app includes two main pages:

### Environmental Dashboard

Loads recent environmental datasets, joins them with world geometry, and lets the user explore country-level patterns through maps, charts, filters, and comparisons.

### AI Workflow

Lets the user choose latitude, longitude, and zoom, download a matching ESRI World Imagery image, describe that image with a local vision model in Ollama, and then assess whether the area appears to be at environmental risk using a second local language model.

The AI workflow is governed by `models.yaml`, and every run is logged in `database/images.csv`. If the same coordinates, zoom, and governed settings were already used before, the app reuses the cached image and stored results instead of running the full pipeline again.

## Repository Structure

```text
app/
  ai_workflow.py         AI pipeline, caching, database logging, ESRI image download
  okavango.py            Dataset download and processing logic
  streamlit_app.py       Main Streamlit application
database/
  images.csv             Persistent log of AI workflow runs
downloads/               Downloaded OWID datasets and Natural Earth map files
images/                  Saved ESRI imagery and example outputs
models.yaml              Governed Ollama models, prompts, and settings
requirements.txt         Python dependencies
tests/                   Automated tests
```

## Requirements

Before installation, make sure you have:

- Python 3.10 or newer
- Internet access for the initial dataset download
- Internet access the first time Ollama needs to pull a model

For the basic dashboard and test suite, Python and the project dependencies are enough.

For the full AI workflow, you also need:

- Ollama installed locally
- The Ollama application or service running
- Enough local hardware resources to run the selected models in `models.yaml`

## Installation

The project was prepared and tested with Windows PowerShell commands. It may also work on other systems, but those users will need to adapt the virtual-environment activation, environment-variable, and path syntax.

1. Clone the repository.
2. Open a terminal in the project root.
3. Create and activate a virtual environment.
4. Install the Python dependencies.
5. Install Ollama only if you want to use the AI workflow.

### Python Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your system uses the Windows launcher, you can use `py` instead of `python` in the commands above.

### Ollama Setup For The AI Workflow

Install Ollama from: <https://ollama.com/download>

After installing it, make sure the Ollama application or service is running locally. By default, the app expects:

```text
http://127.0.0.1:11434
```

If your Ollama server is running elsewhere, set one of these environment variables before starting the app:

```powershell
$env:OLLAMA_BASE_URL="http://127.0.0.1:11434"
```

or

```powershell
$env:OLLAMA_HOST="127.0.0.1:11434"
```

The app will automatically pull missing models the first time they are needed.

## How To Run

From the project root, run either:

```powershell
python -m streamlit run app\streamlit_app.py
```

or

```powershell
py -m streamlit run app\streamlit_app.py
```

Then open the local Streamlit URL shown in the terminal.

## How To Use The AI Workflow

1. Open the `AI workflow` page from the sidebar.
2. Choose a location either from the built-in country and city list or by entering custom coordinates.
3. Select a zoom level.
4. Click `Run AI workflow`.

The app will then:

1. Download an ESRI World Imagery image into `images/`.
2. Read the governed model and prompt settings from `models.yaml`.
3. Generate an image description with the configured vision model.
4. Ask a second model to assess environmental risk from that description.
5. Show the image, description, risk badge, evidence, and follow-up questions.
6. Append the full result to `database/images.csv`.
7. Reuse a cached result later if the same settings are selected again.

## Governed AI Configuration

The file `models.yaml` stores:

- The image-analysis model
- The image-analysis prompt
- Image settings such as temperature and image size
- The text-analysis model
- The text-analysis prompt
- Text-generation settings

This keeps the workflow reproducible and makes it possible to explain which exact prompts and models produced each logged result.

## Verification

You can verify that the project is set up correctly by running the test suite from the project root.

The tests validate core workflow behavior such as dataset download logic, world-map merging, configuration loading, cache handling, and structured AI risk output behavior.

Run either:

```powershell
python -m pytest -q
```

or

```powershell
py -m pytest -q
```

## Example Environmental Risk Detections

Below are three saved examples from the app's AI workflow. These are prototype outputs generated by the application and should not be treated as validated environmental assessments.

### Example 1: Cairo, Egypt

![Cairo example](images/esri_30.0444_31.2357_z14.jpg)

- Coordinates: `30.0444, 31.2357`
- Zoom: `14`
- Risk result: `high` with score `90`
- Flagged: `Y`
- Summary: Visible signs of deforestation, urbanization, and water bodies suggest a high level of environmental concern.

### Example 2: Lisbon, Portugal

![Lisbon example](images/esri_38.7223_-9.1393_z14.jpg)

- Coordinates: `38.7223, -9.1393`
- Zoom: `14`
- Risk result: `high` with score `92`
- Flagged: `Y`
- Summary: Visible signs of deforestation, land degradation, and urban encroachment.

### Example 3: Coimbra, Portugal

![Coimbra example](images/esri_40.2033_-8.4103_z14.jpg)

- Coordinates: `40.2033, -8.4103`
- Zoom: `14`
- Risk result: `medium` with score `68`
- Flagged: `Y`
- Summary: Moderate environmental concerns, but no clear signs of severe stressors.

## Why This Project Can Help The UN SDGs

Project Okavango is a prototype, but it shows how low-cost data tools and local AI workflows can support environmental monitoring in a practical way. By combining recent public datasets with satellite imagery, the app can help users quickly identify patterns that deserve deeper investigation. It does not replace expert analysis, but it can reduce the time needed to move from raw data to a first environmental risk signal.

The project is especially connected to these Sustainable Development Goals:

- SDG 13: Climate Action
- SDG 15: Life on Land
- SDG 11: Sustainable Cities and Communities
- SDG 6: Clean Water and Sanitation

In practice, a tool like this could help NGOs, municipalities, researchers, or student teams prioritize where to look next by flagging locations for deeper human review.

## Troubleshooting

- If Streamlit does not start, make sure the virtual environment is activated and the dependencies from `requirements.txt` were installed successfully.
- If GeoPandas fails during setup, reinstall the Python dependencies inside a clean virtual environment and confirm you are using a supported Python version.
- If the AI workflow fails, make sure Ollama is installed, running locally, and reachable at the configured `OLLAMA_BASE_URL` or `OLLAMA_HOST`.
- If a model call fails, check whether Ollama still needs to pull the model listed in `models.yaml` or whether the machine has enough memory for that model.

## Notes

- The app is a proof of concept, so model outputs may vary depending on machine performance and available memory.
- The first Ollama run may take longer because models may need to be downloaded.
- The AI workflow uses free tools and public imagery, so the goal is technical functionality and reproducibility rather than perfect environmental diagnosis.
- Reproducibility is limited by the local Ollama model version, available hardware, and the prompts and settings defined in `models.yaml`.
