# Backend image used by docker-compose.yml.
#
# The VS Code extension does not use this: it provisions its own virtual
# environment on the host (extension/src/pythonEnv.ts). This image is for
# driving the HTTP API directly.
FROM python:3.12-slim

WORKDIR /app

# Dependencies first, so editing backend code does not re-run pip.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend

ENV CODELENS_INDEX_PATH=/app/.codelens_index
EXPOSE 8000

# 0.0.0.0 inside the container so the published port can reach it;
# docker-compose.yml publishes that port on the host's loopback only.
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
