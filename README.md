# Academic Research Assistant

An evidence-based AI application that answers questions from uploaded academic documents and provides page-level citations.

## Current Features

- FastAPI backend
- Health-check API endpoint
- PDF text extraction
- Page-number and filename preservation
- Overlapping text chunking
- Automated tests with Pytest

## Project Goal

The finished application will allow users to upload academic PDFs, ask questions, receive evidence-based answers, and view the supporting document pages.

## Run the Tests

    python -m pytest -q

## Run the API

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload