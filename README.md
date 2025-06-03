# Markov-AI-Trends
Final project developed during the Discrete Mathematics course 2025-01 using Python

Purpose and Scope:
This document provides a comprehensive overview of the Markov-AI-Trends system, a Python-based application that analyzes user navigation patterns between AI platforms using Markov chain mathematics. The system transforms user platform preference data into predictive models that forecast AI platform adoption trends.

This page covers the system's fundamental purpose, architecture, and key components. For detailed implementation specifics, see Core Implementation. For mathematical theory details, see Mathematical Foundation. For practical usage instructions, see Usage Guide.

System Purpose and Functionality:
Markov-AI-Trends is an educational discrete mathematics project that demonstrates real-world applications of Markov chain theory. The system processes user navigation sequences between five major AI platforms and generates statistical predictions about platform switching behaviors and long-term adoption trends.

The application serves dual purposes:
- Academic: Demonstrates discrete mathematics concepts including stochastic matrices, eigenvalue analysis, and probability theory
- Analytical: Provides insights into AI platform usage patterns and transition probabilities

Core Functionality:
- Load and validate user platform navigation data from CSV files
- Calculate transition probability matrices with Laplace smoothing
- Generate n-step predictions using matrix exponentiation
- Analyze long-term equilibrium states through stationary distributions
- Provide platform recommendations based on transition probabilities
- Present results through an interactive console interface

System Architecture:
The following diagram illustrates the high-level architecture connecting data sources to analytical outputs through the core Markov processing engine:

System Component Architecture:
![image](https://github.com/user-attachments/assets/c4757d6b-b021-44b0-a4de-da10d577ab4f)

This architecture separates concerns into three distinct layers: data management, mathematical processing, and user interaction. The core engine implements Markov chain algorithms while the interface layer provides educational access to the mathematical models.

Data Processing Pipeline:
The system follows a structured pipeline that transforms raw user navigation data into mathematical models for trend analysis.

Data Flow and Mathematical Transformation Pipeline:
![image](https://github.com/user-attachments/assets/62dd1fda-aede-4fd4-aeb5-c27dcb7e9da4)

The pipeline implements proper stochastic matrix operations, ensuring mathematical validity through normalization and smoothing techniques. Each stage validates data integrity before passing results to subsequent processing steps.

Sources: High-level system diagrams, data flow analysis

Key System Components:
The system consists of several interconnected modules that handle different aspects of the Markov chain analysis

| Component | Function Name | Purpose |
|----------|----------|----------|
| Data Loader | obtener_datos_ia_plataformas() | Parse CSV data and validate sequences |
| Transition Calculator | calcular_matriz_transicion() | Generate stochastic transition matrix |
| Prediction Engine | calcular_matriz_orden_n() | Compute n-step transition probabilities |
| Long-term Analyzer | pronosticar_largo_plazo() | Calculate stationary distribution |
| Recommendation System | generar_recomendaciones() | Identify high-probability transitions |
| User Interface | main() menu system | Provide interactive console access |

The system maintains mathematical rigor through proper implementation of discrete mathematics concepts while providing an accessible interface for exploring Markov chain applications.

Sources: High-level system diagrams, function analysis

Mathematical Foundation
The system implements core concepts from discrete mathematics and probability theory:

- Stochastic Matrices: 5×5 transition probability matrices representing platform switching behaviors
- Markov Property: Memoryless state transitions where future states depend only on current state
- Matrix Exponentiation: Computing P^n for multi-step predictions using linear algebra
- Eigenvalue Analysis: Finding stationary distributions through dominant eigenvector calculation
- Laplace Smoothing: Statistical technique to handle zero probabilities in sparse data

For detailed mathematical theory and implementation specifics, see Mathematical Foundation.

Sources: High-level system diagrams, mathematical model analysis

User Interaction Model:
The system provides educational access through a console-based menu interface that allows exploration of different analytical perspectives.

The interface prioritizes educational clarity, allowing users to examine the mathematical model from multiple perspectives while maintaining focus on discrete mathematics learning objectives.

Sources: High-level system diagrams, user interaction analysis

Project Context:
Markov-AI-Trends serves as a final project for a Discrete Mathematics course (2025-01), demonstrating practical applications of theoretical concepts in real-world data analysis scenarios. The system balances academic rigor with practical implementation, using standard scientific Python libraries while maintaining focus on mathematical fundamentals.
