from fastapi import FastAPI

app = FastAPI(
    title="Academic Research Assistant",
    description="An evidence-based document question-answering application.",
    version="0.1.0",
)


@app.get("/")
def read_root() -> dict[str, str]:
    """Return basic information about the application."""
    return {
        "name": "Academic Research Assistant",
        "version": "0.1.0",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """Confirm that the API is operating."""
    return {"status": "healthy"}
