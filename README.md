# 🤖 DataPilot AI

> An AI-powered autonomous data analysis agent for cleaning, analyzing, visualizing, and explaining datasets.

## 📌 Overview

**DataPilot AI** is a Python and Streamlit-based data analysis application that allows users to upload CSV or Excel datasets and interact with their data using natural language.

The application can inspect dataset quality, identify duplicates and missing values, perform numerical and categorical analysis, calculate correlations, generate visualizations, and produce a complete analytical report.

It also includes an autonomous analysis workflow that can perform multiple analysis steps automatically.

## ✨ Key Features

- 📂 CSV and Excel dataset upload
- 🔍 Automatic dataset profiling
- 📊 Dataset overview and column information
- 🧹 Duplicate detection and removal
- 🩹 Missing-value detection and handling
- 📈 Numerical data analysis
- 🏷️ Categorical and grouped analysis
- 🔗 Correlation analysis
- 📊 Histogram visualization
- 📊 Bar chart visualization
- 🔵 Scatter plot visualization
- 🔥 Correlation heatmap
- 🤖 Natural-language AI assistant
- 🧠 Autonomous multi-step data analysis
- 📝 AI-generated analysis report
- 🔎 Agent activity / decision trace
- 📥 Cleaned dataset export
- 📄 Report export
- 📴 Local/offline fallback for core analysis when AI API access is unavailable

## 🧠 Autonomous Analysis Workflow

DataPilot AI can work through an analysis pipeline instead of requiring the user to perform every step manually.

```text
Upload Dataset
      ↓
Inspect Dataset
      ↓
Check Data Quality
      ↓
Detect Duplicates
      ↓
Handle Missing Values
      ↓
Analyze Numerical Data
      ↓
Analyze Categorical Data
      ↓
Calculate Correlations
      ↓
Select Relevant Visualizations
      ↓
Generate Insights
      ↓
Create Complete Analysis Report
      ↓
Export Results
```

## 🏗️ Project Architecture

```text
DataPilot-AI/
│
├── Data/
│   └── sample.csv
│
├── src/
│   ├── agent_tools.py
│   ├── ai_agent.py
│   ├── data_analyzer.py
│   ├── data_cleaner.py
│   ├── data_exporter.py
│   ├── data_loader.py
│   ├── data_profiler.py
│   └── visualizer.py
│
├── tests/
│   ├── test_ai.py
│   └── test_engine.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application and analysis logic |
| Streamlit | Interactive web dashboard |
| Pandas | Data loading, cleaning, and analysis |
| Plotly | Interactive data visualizations |
| OpenAI API | Natural-language AI capabilities |
| python-dotenv | Environment variable management |
| OpenPyXL | Excel file support |

## 📊 Analysis Capabilities

DataPilot AI supports:

### Data Quality
- Dataset dimensions
- Column types
- Missing-value counts
- Duplicate-row detection
- Duplicate removal

### Numerical Analysis
- Descriptive statistics
- Minimum and maximum values
- Mean and median
- Range
- Column comparisons
- Grouped averages

### Categorical Analysis
- Category information
- Grouped numerical analysis
- Highest and lowest group averages

### Relationship Analysis
- Correlation calculation
- Numerical relationship exploration
- Scatter plots
- Correlation heatmaps

## 💬 Example Questions

After uploading a dataset, users can ask questions such as:

```text
Remove duplicate rows.

What is the average salary?

Which department has the highest average salary?

What is the difference between the highest and lowest salary?

Show the relationship between age and salary.

Which numerical columns are correlated?

Give me a complete analysis of this dataset.
```

## 📈 Sample Dataset

The repository includes a small sample dataset containing fields such as:

- Name
- Age
- Salary
- Department

The sample data can be used to test the application's cleaning, analysis, visualization, and AI features.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/ashfiya015-stack/DataPilot-AI.git
cd DataPilot-AI
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\\Scripts\\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 🔐 OpenAI API Configuration

Create a local `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

**Never upload the `.env` file or your API key to GitHub.**

The repository's `.gitignore` is configured to exclude environment files and other local files.

## ▶️ Run the Application

Start the Streamlit application with:

```bash
python -m streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## 🧪 Testing

The project includes separate test scripts for the analysis engine and AI functionality.

Run the engine tests:

```bash
python tests/test_engine.py
```

Run the AI test:

```bash
python tests/test_ai.py
```

## 📥 Export

DataPilot AI provides downloadable outputs including:

- Cleaned dataset
- Analysis report

## 📴 Fallback Analysis

If AI API access is unavailable, the application can use local analysis logic for core data-analysis tasks so that the application can still provide useful analytical results.

## 🎯 Project Objective

Traditional exploratory data analysis often requires a sequence of repetitive manual tasks:

```text
Load → Inspect → Clean → Analyze → Visualize → Interpret
```

DataPilot AI combines these steps into an interactive workflow:

```text
Upload → Ask / Analyze → DataPilot AI → Insights + Visualizations + Report
```

The goal is to make exploratory data analysis more accessible while keeping the underlying analysis transparent and reviewable.

## 🔮 Future Improvements

Potential future improvements include:

- Database connectivity
- Support for additional file formats
- Advanced anomaly detection
- Automated machine-learning model selection
- More visualization types
- Cloud deployment
- Multi-dataset analysis
- Persistent analytical memory
- Role-based dashboards

## ⚠️ Limitations

- Results depend on the quality and structure of the uploaded dataset.
- AI-generated explanations should be reviewed by the user.
- OpenAI-powered features require a valid API configuration.
- Very large datasets may require additional optimization.

## 👩‍💻 Author

**Ashfiya**

GitHub: https://github.com/ashfiya015-stack

---

⭐ If you find DataPilot AI useful, consider giving the repository a star.
