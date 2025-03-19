FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar archivos de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p data logs backups

# Exponer puerto para health check
EXPOSE $PORT

# Comando por defecto
CMD ["python", "main.py", "run", "--symbol", "ETHUSDT", "--interval", "1h"]
