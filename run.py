import uvicorn

if __name__ == "__main__":
    # Esto le dice a Uvicorn que busque la aplicación "app" dentro de backend/app/main.py
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
