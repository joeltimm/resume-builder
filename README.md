# AI Resume Builder

A web application that uses AI to help tailor a resume to a specific job description quickly and efficiently.

## Core Features

- **Job Description Analysis**: Paste a job description to be analyzed.
- **AI-Powered Matching**: Automatically selects the most relevant skills and accomplishments from a user-managed database.
- **Data Management**: A simple interface to add, edit, and manage your personal skills and work history.
- **Review & Edit**: A final review step to approve or modify the AI's selections.
- **PDF Export**: Generate a polished, professional resume in PDF format.

## Technology Stack

| Component | Technology | Why it's used |
| :--- | :--- | :--- |
| **Backend** | Python, Flask | A simple and powerful framework for creating the API. |
| **Frontend** | HTML, Tailwind CSS, JavaScript | For a clean, responsive, and interactive user interface. |
| **Database** | PostgreSQL with `pgvector` | A robust database with vector support for efficient AI similarity searches. |
| **AI** | `sentence-transformers` | A Python library to convert text into numerical embeddings for semantic matching. |
| **Operations** | Docker, Docker Compose | To containerize the application for consistent, reliable deployment. |
| **PDF Generation** | Stirling-PDF | A Dockerized, API-driven tool for converting HTML to PDF. |

## How to Run

This project is fully containerized using Docker.

### Prerequisites

- Docker
- Docker Compose

### Steps

1. **Clone the repository:**

```bash
    git clone [https://github.com/joeltimm/resume-builder.git](https://github.com/joeltimm/resume-builder.git)
    cd resume-builder
    ```

2. **Create an environment file:**
    Create a file named `.env` in the root of the project and add the following variables. Replace `your_secure_password` with a real password.

```env
    # .env
    POSTGRES_USER={USER}
    POSTGRES_PASSWORD=your_secure_password
    POSTGRES_DB=resume_builder_db
    ```

3.**Build and run the application:**

```bash
    docker-compose up -d --build
    ```

4. **Access the application:**
    - **Resume Builder**: `http://<your-server-ip>:8080`
    - **Data Management Page**: `http://<your-server-ip>:8080/manage-data.html`
    - **Backend API**: `http://<your-server-ip>:5031/api`
