# Flask AHP & TOPSIS Project
![logo](static/logout.png)

This project is a web application built with Flask that performs AHP (Analytic Hierarchy Process) and TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) calculations. It helps in decision-making processes by providing a user-friendly interface to input criteria, sub-criteria, and alternatives, and then calculates the rankings based on these inputs.

## Table of Contents

- [Overview](#overview)
- [Software Architecture](#software-architecture)
- [Frontend](#frontend)
- [Backend](#backend)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Dependencies](#dependencies)
- [Demonstration](#demonstration)
- [Contributing](#contributing)

## Overview

This Flask application facilitates decision-making by implementing the AHP and TOPSIS methods. Users can input criteria and sub-criteria, perform pairwise comparisons, and obtain rankings for various alternatives. The application ensures consistency in the comparisons and provides sensitivity analysis to evaluate the impact of changing criteria weights.

## Software Architecture

The application follows a modular structure with a clear separation of concerns between the frontend and backend. The backend is built with Flask, handling the AHP and TOPSIS calculations and database operations using PostgreSQL. The frontend uses Jinja2 templates for rendering the HTML pages.

## Frontend

### Technologies Used

- HTML
- CSS
- Jinja2

### Frontend Project Structure

- **Templates:** Contains the HTML templates for rendering different pages.
- **Static:** Contains static files such as CSS and JavaScript.

## Backend

### Technologies Used

- Flask
- NumPy
- PostgreSQL
- psycopg2

### Backend Project Structure

The backend code follows a modular and organized structure, leveraging the power of Flask for building a robust and scalable application.

- **app_with_subcriteria.py:** The main application file containing route definitions and core logic.
- **static:** Directory for static files (CSS, JavaScript, images).
- **templates:** Directory for HTML templates.
- **ahp.sql:** SQL script for setting up the database schema.

### Dependencies

1. **Flask:** Web framework for Python.
2. **NumPy:** Library for numerical computations.
3. **psycopg2:** PostgreSQL database adapter for Python.
4. **Jinja2:** Templating engine for Flask.

## Getting Started

### Prerequisites

- Python 3.7+
- PostgreSQL
### Demonstration
Click the link below to watch a demonstration video:

https://github.com/oussama-art/ahp_project/assets/59901157/82e90556-89ea-4ec3-b99c-750b880108e3
### Setup

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd <project_folder>



Configure Database:

Create a PostgreSQL database.
Update the database configuration in app.py.
Set Up Database Schema:

Run the provided SQL script to set up the database schema.
bash
Copy code
psql -U <your_postgres_user> -d <your_database_name> -f path/to/your/schema.sql
Run the Application:

```bash
flask run```
Usage
Register and Login:

Register a new user or log in with existing credentials.
Create Project:

Input the project name and the number of criteria.
Add Criteria and Sub-Criteria:

Define criteria and their respective sub-criteria.
Perform Comparisons:

Perform pairwise comparisons for criteria and sub-criteria.
Calculate Results:

View the AHP and TOPSIS calculation results.
Sensitivity Analysis:

Evaluate the impact of changing criteria weights on the rankings.
###Folder Structure

flask_ahp_topsis/
├── static/
│   ├── logout.png
│   ├── profile.png
│   └── style.css
├── templates/
│   ├── Graph.html
│   ├── add_criteria.html
│   ├── calculate_results.html
│   ├── compare_criteria.html
│   ├── compare_subcriteria.html
│   ├── cri_results.html
│   ├── dashboard.html
│   ├── enter_comparison.html
│   ├── history.html
│   ├── home.html
│   ├── index.html
│   ├── layout.html
│   ├── login.html
│   ├── profile.html
│   ├── register.html
│   ├── results.html
│   ├── resultstops.html
│   ├── sensitivity.html
│   ├── sub_cri_results.html
│   ├── top-menu.html
│   ├── topsis_results.html
│   ├── topsys.html
│   └── topsyss.html
├── .vscode/
├── app_with_subcriteria.py
├── designer_graph.png
├── test.py
├── ahp.sql
└── requirements.txt
###Dependencies

Flask==2.0.2
NumPy==1.21.2
psycopg2==2.9.1
Jinja2==3.0.1
###Contributing
We welcome contributions from everyone. If you would like to contribute, please follow these guidelines:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes and commit them (git commit -m 'Add new feature').
Push to the branch (git push origin feature-branch).
Create a new Pull Request.



