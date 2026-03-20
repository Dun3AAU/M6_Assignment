## Assignment: Automated Weather Pipeline with GitHub Pages

**Deadline:**  
Sunday **22 March 2026 – 20:00 (Danish time)**

The goal of this assignment is to build a **small automated data pipeline** that collects weather forecasts, stores them in a database, generates a creative text output using an LLM, and publishes the result using **GitHub Pages**.

This assignment combines several elements from the course:

- APIs
- Data pipelines
- SQL databases
- LLM usage
- GitHub Actions automation
- GitHub Pages publishing

---

### 1. Create a New GitHub Repository

Create a **new public GitHub repository** for this assignment.


Your repository will contain:

- Python scripts
- a SQL database
- a GitHub Action workflow
- a GitHub Pages website

---

### 2. Collect Weather Data

Use the **Open-Meteo API** to collect weather forecast data.

API documentation:

https://open-meteo.com/

You must fetch **weather data for the next day** for the following locations:

1. **Your place of birth**
2. **Your last residence before arriving to Aalborg**
3. **Aalborg**

(If all 3 are Aalborg, use one as Copenhagen and one as Nice (FR))

---

### 3. Weather Variables

Use **at least three weather variables**, for example:

- temperature
- precipitation
- wind speed
- cloud cover
- humidity

You may choose which variables you want to use.

---

### 4. Create the Data Collection Script

Create a Python script:
**fetch.py**

The script should:

1. Call the **Open-Meteo API**
2. Retrieve forecast data for the three locations
3. Extract the selected weather variables
4. Store the data in a **SQL database**

---

### 5. Store the Data in a SQL Database

Create a local database in your repository.

The database should contain at least:

- location name
- forecast date
- selected weather variables

You may implement the database writing **directly inside `fetch.py`**.

---

### 6. Generate a Poem Using Groq

Use the **Groq API** to generate a short poem.

The poem should:

- compare the weather in the three locations
- describe the differences
- suggest where it would be **nicest to surf tomorrow**
- be written in **two languages**

Example idea: English + your native language


The poem should be generated automatically after the weather data is fetched.

---

### 7. Automate the Pipeline with GitHub Actions

Create a **GitHub Actions workflow** that runs the pipeline automatically.

The workflow should:

1. Run the weather data script
2. Generate the poem
3. update the results

Schedule the workflow to run:
**Every day at 20:00 Danish time**


Use a **cron schedule** in your workflow file.

---

### 8. Publish the Result with GitHub Pages

Create a **GitHub Pages site** inside your repository.

Your site should display:

- the generated poem
- optionally the weather information

You may:

- generate the page automatically in the GitHub Action
- update `docs/index.html`

---

### 9. Submission

Keep the repository **public**.

Submit by:

1. Pushing your final solution to GitHub
2. Sending me the **repository link via Teams**

---

### Goal of the Assignment

By completing this assignment you will build a small **end-to-end MLOps pipeline**:
