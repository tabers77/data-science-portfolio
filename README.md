# Data Science Portfolio - Carlos De La Cruz

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine_Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://github.com/tabers77)
[![Deep Learning](https://img.shields.io/badge/Deep_Learning-00ADD8?style=for-the-badge&logo=pytorch&logoColor=white)](https://github.com/tabers77)
[![NLP](https://img.shields.io/badge/NLP-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://github.com/tabers77)

Welcome to my Data Science portfolio! This repository showcases my expertise in machine learning, deep learning, natural language processing, and data-driven solutions for real-world business problems.

## 👨‍💻 About Me

I'm a Data Scientist specializing in advanced machine learning techniques, natural language processing, and time series forecasting. With hands-on experience in deploying end-to-end ML pipelines, I focus on building scalable, production-ready solutions that drive measurable business impact.

**Key Expertise:**
- 🤖 Machine Learning & Deep Learning (LSTM, BERT, XGBoost, Random Forest)
- 📊 Natural Language Processing & Topic Modeling (LDA, BERTopic, Transformers)
- 📈 Time Series Forecasting & Demand Prediction
- 🎯 Recommender Systems (Hybrid Collaborative & Content-Based)
- 🧪 LLM/Agent Evaluation & Prompt Optimization (Custom Frameworks, DeepEval, Ragas)
- 🔬 Experiment Tracking & MLOps (MLflow, Databricks)
- 💾 Big Data Processing (PySpark, Snowflake)

## 📫 Contact

- **Email:** [tabers77@gmail.com](tabers77@gmail.com) 
- **LinkedIn:** [https://www.linkedin.com/in/carlosdlc/](https://www.linkedin.com/in/carlosdlc/) 
- **Medium Blog:** [@tabers77](https://medium.com/@tabers77)
- **GitHub:** [@tabers77](https://github.com/tabers77)

---

## 📂 Table of Contents

- [About Me](#-about-me)
- [Featured Projects](#-featured-projects)
  - [Hybrid Topic Modeling System](#1-hybrid-topic-modeling-system)
  - [AI-Powered Retail Optimization](#2-ai-powered-retail-optimization)
- [Experimental Projects](#-experimental-projects)
  - [Nested Learning - Continual Learning Research](#-nested-learning---continual-learning-research)
  - [Evallab - LLM/Agent Evaluation Framework](#-evallab---llmagent-evaluation-framework)
  - [LLM Experiments - RAG Chatbot System](#-llm-experiments---rag-chatbot-system)
- [Skills & Technologies](#-skills--technologies)
- [Contact](#-contact)

---

## 🚀 Featured Projects

### 1. 🔬 Hybrid Topic Modeling System

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat&logo=github)](https://github.com/tabers77/Hybrid-Topic-Modeling-System)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](https://opensource.org/licenses/MIT)

**An intelligent NLP system for workplace alert classification using advanced topic modeling techniques.**

#### 🎯 Project Overview

Developed a comprehensive text classification system that automatically categorizes 3,000+ workplace alerts using multiple machine learning approaches. This project demonstrates a rigorous comparative analysis of traditional and modern NLP techniques, revealing surprising insights about model performance.

#### 🔑 Key Highlights

- **Multi-Model Comparison:** Implemented and evaluated LDA, K-means, BERT, and novel Hybrid BERT-LDA approaches
- **Surprising Finding:** LDA achieved **2x better coherence** (0.7765) compared to BERT (0.3970), challenging the "bigger is better" paradigm in modern NLP
- **Custom Architecture:** Developed hybrid BERT-LDA model combining semantic understanding with interpretable topic distributions
- **Outlier Detection:** Implemented probability-based threshold system reducing outliers from 39.3% (BERT) to ~11% (LDA)
- **Advanced Preprocessing:** NER-based entity removal for privacy-preserving text cleaning
- **Production-Ready:** MLflow integration for experiment tracking and model versioning

#### 💡 Business Impact

- ✅ Automated classification of thousands of workplace alerts
- ✅ Discovered 21 meaningful incident categories (PPE, Equipment, Chemical Handling, etc.)
- ✅ Enabled faster pattern recognition and proactive risk management
- ✅ Provides analytical foundation for data-driven operational improvements

#### 🛠️ Technologies Used

`Python` `Scikit-learn` `BERTopic` `Sentence-Transformers` `NLTK` `Pandas` `MLflow` `PySpark` `Snowflake`

#### 📊 Model Performance

| Model | Topics | Coherence Score | Outlier Rate | Status |
|-------|--------|----------------|--------------|--------|
| **LDA** | 21 | **0.7765** | ~11% | ✅ **Best** |
| BERT | 23 | 0.3970 | 39.3% | ❌ |
| Hybrid BERT-LDA | Variable | Combined | Moderate | ⚡ Balanced |
| K-means | 2-8 | N/A | High Imbalance | ❌ |

#### 🔬 Research Contributions

- Published findings on LDA superiority over BERT for domain-specific text classification
- Novel hybrid architecture combining neural and statistical methods
- Framework for privacy-preserving NLP preprocessing

[**→ View Project Repository**](https://github.com/tabers77/Hybrid-Topic-Modeling-System)

---

### 2. 🛒 AI-Powered Retail Optimization

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat&logo=github)](https://github.com/tabers77/ai-powered-retail-optimization)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6+-orange?style=flat&logo=tensorflow)](https://www.tensorflow.org/)

**End-to-end ML system for retail inventory optimization through demand forecasting and personalized recommendations.**

#### 🎯 Project Overview

Built two interconnected machine learning systems to optimize retail operations: a time series forecasting model for demand prediction and a hybrid recommender system for product assortment optimization.

#### 🔑 Key Highlights

**Time Series Forecasting:**
- **LSTM with Attention:** Achieved **RMSE of 10.90**, outperforming XGBoost (11.87) and Random Forest (12.34)
- **Feature Engineering:** Integrated weather data (5-8% error reduction), holiday features (2-3% improvement), and cyclical time encodings
- **Optimal Configuration:** 14-day sliding window, 32 LSTM units, NAdam optimizer
- **Multivariate Approach:** External features improved accuracy by 8-12% over univariate models

**Recommender System:**
- **Hybrid Architecture:** 60% collaborative filtering + 40% content-based filtering
- **Clustering-Based:** Gaussian Mixture Model for cabinet segmentation (60% computation reduction)
- **Quality Metrics:** 25% improved diversity over pure collaborative filtering
- **Scalable Design:** Matrix factorization with SVD for handling sparse user-item matrices

**Experimental Pricing Module (In Development):**
- Thompson Sampling algorithm for dynamic price optimization
- Beta and Gamma distribution modeling for demand uncertainty

#### 💡 Business Impact

- ✅ Accurate demand forecasting reduces waste through precise inventory prediction
- ✅ Personalized recommendations improve product-cabinet matching
- ✅ Data-driven inventory management improves operational efficiency
- ✅ Daily aggregation strategy optimized for real-time decision making

#### 🛠️ Technologies Used

`Python` `TensorFlow/Keras` `XGBoost` `Scikit-learn` `Pandas` `Numpy` `MLflow` `PySpark`

#### 📊 Model Comparison

| Model | RMSE | MAE | Training Time* | Status |
|-------|------|-----|----------------|--------|
| **Multivariate LSTM + Attention** | **10.90** | **7.85** | ~43 min | ✅ **Selected** |
| Univariate LSTM | 11.39 | 8.21 | ~43 min | ❌ |
| XGBoost | 11.87 | 8.45 | ~18 min | ❌ |
| Random Forest | 12.34 | 8.92 | ~15 min | ❌ |

*Training time for 20% of cabinets on standard compute

#### 🔍 Key Insights

- **Optimal Window Size:** 14 days outperformed 7, 21, and 28-day windows
- **Optimizer Selection:** NAdam consistently beat Adam and RMSprop
- **LSTM Architecture:** 32 units optimal (64 units increased error)
- **Feature Importance:** Historical sales > Day of week > Month > Weather > Holidays

[**→ View Project Repository**](https://github.com/tabers77/ai-powered-retail-optimization)

---

## 🧪 Experimental Projects

### 🧠 Nested Learning - Continual Learning Research

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat&logo=github)](https://github.com/tabers77/Machine-Learning-Projects/tree/main/Nested-Learning)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)](https://github.com/tabers77/Machine-Learning-Projects/tree/main/Nested-Learning)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)](https://www.python.org/)

**PyTorch implementation of the Nested Learning paper (NeurIPS 2025) with original research on multi-timescale updates and catastrophic forgetting**

#### 🎯 Project Overview

Implementation of "Nested Learning: The Illusion of Deep Learning Architectures" (Ali Behrouz et al., NeurIPS 2025), featuring an original continual learning experiment exploring how multi-timescale neural network updates affect the stability-plasticity tradeoff. Inspired by biological memory systems — fast working memory (hippocampus) alongside slow consolidated knowledge (neocortex) — the project tests whether making network components change at different speeds mitigates catastrophic forgetting.

#### 🔑 Key Highlights

- **Complementary Memory System (CMS):** Layers updating at geometrically increasing intervals (every 1, 4, 16 steps), mirroring brain oscillation frequencies
- **14% Forgetting Reduction:** CMS reduces catastrophic forgetting compared to naive training while maintaining equivalent plasticity
- **Associative Memory Unification:** Demonstrates that attention, backpropagation, and optimizer momentum all function as associative memory systems
- **Deep Momentum Gradient Descent (DMGD):** Meta-learned optimizer using small neural networks to discover superior momentum functions
- **Hope Architecture:** Full model combining attention-based memory, CMS multi-frequency blocks, and surprise-based gating
- **Comprehensive Testing:** 45 pytest unit tests across all modules

#### 📊 Experiment Results

| Method | Forgetting | Final Accuracy | Plasticity |
|--------|-----------|----------------|-----------|
| Naive | 40.8% | 57.8% | 98.6% |
| EWC | 40.7% | 57.8% | 98.5% |
| **CMS (C=32)** | **35.2%** | **63.0%** | 98.1% |

#### 🔬 Key Findings

- **Monotonic improvement** with timescale separation — forgetting decreased steadily as update intervals increased, plateauing around C=32-64
- **Free stability** — plasticity remained flat across all conditions (97.8-98.6%), fast layers handle new learning while slow layers preserve old knowledge
- **EWC ineffectiveness** — Elastic Weight Consolidation barely outperformed naive training, possibly due to noisy Fisher information estimates

#### 🛠️ Technologies Used

`Python` `PyTorch` `Hopfield Networks` `Continual Learning` `EWC` `MNIST`

#### 💡 Key Learnings

- Implementing Complementary Learning Systems theory (McClelland et al., 1995) in neural network architectures
- Multi-timescale update mechanisms as an alternative to penalty-based continual learning approaches
- Biological plausibility of architectural solutions for catastrophic forgetting
- Experimental design for stability-plasticity tradeoff analysis

[**→ View Project Repository**](https://github.com/tabers77/Machine-Learning-Projects/tree/main/Nested-Learning)

---

### 🧪 Evallab - LLM/Agent Evaluation Framework

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat&logo=github)](https://github.com/tabers77/evallab)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)](https://github.com/tabers77/evallab)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)](https://www.python.org/)

**Framework-agnostic LLM/Agent evaluator with reinforcement learning support**

#### 🎯 Project Overview

A production-grade evaluation framework that converts traces from multi-agent systems (AutoGen, LangGraph) into a canonical trajectory format, runs pluggable scorers, and produces multi-dimensional evaluation results. Outputs feed human-readable reports, RL reward functions, or prompt optimization loops. Built with zero core dependencies and protocol-based design (PEP 544) for maximum extensibility.

#### 🔑 Key Highlights

- **Pipeline Architecture:** Adapters → Episode/Step → Scorers → ScoreVector → Rewards → RL Training
- **Multi-Framework Support:** AutoGen event log parser (custom brace-counting state machine) and LangGraph astream_events adapter
- **6 Built-in Scorers:** Numeric consistency, issue detection, rule-based deduction, LLM-as-Judge, DeepEval (50+ metrics), and Ragas wrappers
- **RL Integration:** HuggingFace TRL (GRPO) bridge, DSPy MIPROv2 optimizer, and generic prompt tuning loop
- **HTTP Reward Server:** FastAPI-based service for distributed RL training
- **Reporting:** Self-contained HTML reports (inline CSS, XSS-escaped), JSON, and terminal text output
- **Zero Core Dependencies:** Entire evaluation pipeline works without optional packages
- **Comprehensive Testing:** 419 tests across 34 test files (unit, smoke, integration, e2e)

#### 🏗️ Architecture

| Layer | Components | Purpose |
|-------|-----------|---------|
| **Adapters** | AutoGen, LangGraph | Convert framework logs to canonical Episodes |
| **Scorers** | Numeric, Rules, LLM-Judge, DeepEval, Ragas | Multi-dimensional evaluation |
| **Rewards** | WeightedSum, Deduction, Composite | Convert scores to RL-compatible signals |
| **RL Bridges** | TRL, DSPy, TuningLoop | Plug into training/optimization frameworks |
| **Reporting** | HTML, JSON, Text, Comparison | Human-readable and machine-readable output |

#### 🛠️ Technologies Used

`Python` `FastAPI` `PyTest` `HuggingFace TRL` `DSPy` `DeepEval` `Ragas` `OpenAI` `Azure OpenAI` `Hatchling`

#### 💡 Key Learnings

- Protocol-based composition (PEP 544) as an alternative to inheritance for framework extensibility
- Building zero-dependency cores with optional feature groups for modular packaging
- Bridging evaluation metrics to RL reward signals for prompt optimization
- Robust log parsing for malformed multi-agent traces
- Implementing PPE (Preference Proxy Evaluations) for reward model benchmarking

[**→ View Project Repository**](https://github.com/tabers77/evallab)

---

### 🤖 LLM Experiments - RAG Chatbot System

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat&logo=github)](https://github.com/tabers77/llms-experiments)
[![Status](https://img.shields.io/badge/Status-Archived-inactive?style=flat)](https://github.com/tabers77/llms-experiments)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)](https://www.python.org/)

**⚠️ Archived/Experimental Project** | *Focus: Rapid deployment of intelligent Q&A chatbots*

#### 🎯 Project Overview

Experimental framework for quickly deploying intelligent question-answering chatbots using Langchain and Azure OpenAI. This project demonstrates rapid prototyping capabilities for RAG-based conversational AI systems with production-ready containerized deployment.

#### 🔑 Key Highlights

- **Quick Deployment Pipeline:** Flask-based web application with Docker containerization for rapid deployment to Azure Web Apps
- **RAG Implementation:** Retrieval-Augmented Generation using Langchain with custom prompt templates
- **Embedding Management:** PostgreSQL-based vector storage for document embeddings and metadata
- **Multi-Agent Architecture:** Experimented with sequential chains, agents, and custom tools
- **Azure Integration:** Full integration with Azure ChatOpenAI and Azure OpenAI embeddings
- **Modular Design:** Configurable system supporting multiple use cases (legal documents, patents, etc.)

#### 🛠️ Technologies Used

`Python` `Langchain` `Azure OpenAI` `Flask` `PostgreSQL` `Docker` `Azure Web Apps` `HTML/CSS/JS`

#### 💡 Key Learnings

- Framework for rapid chatbot prototyping and deployment
- Hands-on experience with RAG chains and memory management
- Production deployment patterns using Docker containers
- Integration patterns for Azure cloud services with LLMs

[**→ View Project Repository**](https://github.com/tabers77/llms-experiments)

---

## 🛠️ Skills & Technologies

### Programming Languages
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)

### Machine Learning & Deep Learning
- **Traditional ML:** Scikit-learn, XGBoost, Random Forest, Ensemble Methods
- **Deep Learning:** TensorFlow, Keras, LSTM Networks, Attention Mechanisms
- **NLP:** BERT, BERTopic, Sentence Transformers, NLTK, spaCy
- **LLM Evaluation:** DeepEval, Ragas, LLM-as-Judge, Custom Scoring Frameworks
- **Topic Modeling:** LDA (Latent Dirichlet Allocation), K-means, HDBSCAN
- **Time Series:** ARIMA, LSTM, Prophet, Exponential Smoothing

### Big Data & MLOps
- **Data Processing:** Pandas, NumPy, PySpark
- **Data Warehouses:** Snowflake, Databricks
- **Experiment Tracking:** MLflow
- **APIs & Services:** FastAPI, Flask, Docker
- **Version Control:** Git, GitHub

### Data Visualization
- **Libraries:** Matplotlib, Seaborn, Plotly
- **Dashboards:** Streamlit, Dash

### Techniques & Methodologies
- Supervised & Unsupervised Learning
- Natural Language Processing
- Time Series Forecasting
- Recommender Systems (Collaborative & Content-Based Filtering)
- Feature Engineering & Selection
- Hyperparameter Optimization (Grid Search, Bayesian Optimization)
- Model Evaluation & Validation
- LLM/Agent Evaluation & RL-based Prompt Optimization
- A/B Testing & Experimentation

---

## 📈 What Sets My Work Apart

1. **Rigorous Experimentation:** Systematic comparison of multiple approaches with comprehensive evaluation metrics
2. **Surprising Insights:** Willingness to challenge conventional wisdom (e.g., LDA outperforming BERT)
3. **Production Focus:** Code structured for deployment with experiment tracking and reproducibility
4. **Business Impact:** Solutions designed to solve real-world problems with measurable outcomes
5. **Hybrid Approaches:** Combining traditional and modern techniques for optimal results

---

## 🎓 Continuous Learning

I regularly share insights and learnings on my [Medium blog](https://medium.com/@tabers77), covering topics in:
- Machine Learning best practices
- NLP techniques and applications
- Time series forecasting strategies
- MLOps and deployment considerations

---

## 📬 Let's Connect!

I'm always interested in discussing data science projects, collaborations, or opportunities. Feel free to reach out!

- 📧 Email: [tabers77@gmail.com](tabers77@gmail.com)
- 💼 LinkedIn: [https://www.linkedin.com/in/carlosdlc/](https://www.linkedin.com/in/carlosdlc/)
- 📝 Medium: [@tabers77](https://medium.com/@tabers77)
- 🐙 GitHub: [@tabers77](https://github.com/tabers77)

---

**⭐ If you find my work interesting, please consider starring the repositories!**

---

*Last Updated: February 2026*