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
- 🔬 Experiment Tracking & MLOps (MLflow, Databricks)
- 💾 Big Data Processing (PySpark, Snowflake)

## 📫 Contact

- **Email:** [your.email@example.com](mailto:your.email@example.com) *(Update with your email)*
- **LinkedIn:** [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile) *(Update with your LinkedIn)*
- **Medium Blog:** [@tabers77](https://medium.com/@tabers77)
- **GitHub:** [@tabers77](https://github.com/tabers77)

---

## 📂 Table of Contents

- [About Me](#-about-me)
- [Featured Projects](#-featured-projects)
  - [Hybrid Topic Modeling System](#1-hybrid-topic-modeling-system)
  - [AI-Powered Retail Optimization](#2-ai-powered-retail-optimization)
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
- ✅ Reduced manual categorization effort by 80%+

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
- ✅ Personalized recommendations increase sales by 15-20% through better product-cabinet matching
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

## 🛠️ Skills & Technologies

### Programming Languages
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![R](https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)

### Machine Learning & Deep Learning
- **Traditional ML:** Scikit-learn, XGBoost, Random Forest, Ensemble Methods
- **Deep Learning:** TensorFlow, Keras, LSTM Networks, Attention Mechanisms
- **NLP:** BERT, BERTopic, Sentence Transformers, NLTK, spaCy
- **Topic Modeling:** LDA (Latent Dirichlet Allocation), K-means, HDBSCAN
- **Time Series:** ARIMA, LSTM, Prophet, Exponential Smoothing

### Big Data & MLOps
- **Data Processing:** Pandas, NumPy, PySpark
- **Data Warehouses:** Snowflake, Databricks
- **Experiment Tracking:** MLflow
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

*Last Updated: January 2026*