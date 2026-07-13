FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .
ENV TRUSTMESH_DB=/data/trustmesh.db
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "trustmesh.app:app", "--host", "0.0.0.0", "--port", "8000"]
