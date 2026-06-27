<div align="center">

# 📚 Agentic Research Paper Summarizer

### Automate literature reviews using a collaborative AI agent pipeline

An **Agentic AI-powered research assistant** that retrieves research papers from **arXiv**, analyzes them through specialized AI agents, evaluates research quality, detects contradictions, and generates comprehensive literature reviews using **local Large Language Models (LLMs)**.

<p align="center">
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/LangGraph-Agentic%20Workflow-blue?style=for-the-badge"/>
<img src="https://img.shields.io/badge/LangChain-Framework-green?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/LLM-Local%20Models-success?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

</div>

---

## 📸 Demo

<!-- > Replace these with your own screenshots or GIF.

| Home Page | Generated Literature Review |
|-----------|-----------------------------|
| ![](assets/home.png) | ![](assets/output.png) |

--- -->

# 📖 Overview

Conducting a literature review often requires reading dozens of research papers, comparing methodologies, identifying conflicting findings, and synthesizing the overall state of the art.

This project automates that workflow through an **agentic architecture** where specialized AI agents collaborate to perform each stage of the research analysis process.

Starting from a research query, the system retrieves relevant papers from **arXiv**, extracts their contents, generates structured summaries, evaluates research quality, identifies contradictions across studies, and finally produces a synthesized literature review.

The application is designed to run locally with configurable Large Language Models, ensuring privacy while remaining flexible enough to support different local inference backends.

---

# ✨ Features

- 🔎 Retrieve research papers directly from arXiv
- 📥 Automatic PDF downloading
- 📄 Intelligent PDF parsing
- 🤖 AI-powered structured summarization
- ⭐ Research quality assessment
- ⚖️ Cross-paper contradiction detection
- 📚 Literature review synthesis
- ⚡ Sequential and parallel execution
- 💾 Markdown & JSON export
- 🔒 Local-first architecture using configurable LLMs

---

# 🏗️ System Workflow

```text
                    Research Query
                          │
                          ▼
              🔎 Retrieve Papers (arXiv)
                          │
                          ▼
                  📥 Download PDFs
                          │
                          ▼
                 📄 PDF Parsing Agent
                          │
                          ▼
             🤖 Summarization Agent
                          │
                          ▼
          ⭐ Quality Assessment Agent
                          │
                          ▼
      ⚖️ Contradiction Detection Agent
                          │
                          ▼
      📚 Literature Review Agent
                          │
                          ▼
        📦 Markdown & JSON Reports
```

---

# 🤖 Agent Pipeline

| Agent | Responsibility |
|--------|----------------|
| 📥 Paper Retrieval | Searches and downloads papers from arXiv |
| 📄 PDF Parser | Extracts clean text from downloaded PDFs |
| 🤖 Summarizer | Produces structured summaries of each paper |
| ⭐ Quality Evaluator | Scores novelty, rigor, clarity and impact |
| ⚖️ Contradiction Detector | Identifies agreements and conflicting findings |
| 📚 Literature Review Generator | Produces a unified review from all analyzed papers |

---

# 🛠 Technology Stack

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| UI | Streamlit |
| Agent Framework | LangGraph |
| LLM Framework | LangChain |
| Research Source | arXiv API |
| PDF Processing | PyMuPDF |
| Models | Local LLMs |

---

# 📂 Project Structure

```text
research-summarizer/
│
├── agents/             # Specialized AI agents
├── graph/              # LangGraph workflows
├── prompts/            # Prompt templates
├── utils/              # Utilities & model configuration
├── cache/              # Downloaded papers
├── outputs/            # Generated literature reviews
├── tests/
│
├── app.py              # Streamlit application
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/mirada725/research-summarizer.git
cd research-summarizer
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ⚙️ Configuration

Configure your preferred local model in

```text
utils/model_config.py
```

Ensure your local inference server (e.g., Ollama) is running before launching the application.

---

# ▶️ Usage

Launch the application

```bash
streamlit run app.py
```

Then

1. Enter a research topic.
2. Choose the number of papers.
3. Select sequential or parallel execution.
4. Run the workflow.
5. Review generated summaries.
6. Export the synthesized literature review.

---

# 📤 Outputs

The application generates

- Individual paper summaries
- Research quality assessments
- Contradiction reports
- Literature review
- Markdown export
- JSON export

Generated reports are stored inside

```text
outputs/
```

Downloaded papers are cached in

```text
cache/
```

---

# 🌟 Future Improvements

- [ ] IEEE Xplore integration
- [ ] ACM Digital Library integration
- [ ] Semantic Scholar support
- [ ] Citation graph visualization
- [ ] Multi-agent debate
- [ ] Human-in-the-loop review
- [ ] RAG-based long-context memory
- [ ] Multi-language support

---

# 🤝 Contributing

Contributions are welcome!

If you'd like to improve the project:

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Open a Pull Request.

---

# 📄 License

This project is licensed under the MIT License.

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

Made with ❤️ using Python, LangGraph and Local LLMs

</div>