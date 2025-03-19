# health_endpoint.py
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import json
import logging
import time
from datetime import datetime

logger = logging.getLogger("health_endpoint")

class HealthCheckHandler(BaseHTTPRequestHandler):
    # Referencia estática al estado del bot
    bot_status = {
        'status': 'starting',
        'last_update': datetime.now().isoformat(),
        'uptime': 0,
        'start_time': datetime.now().isoformat(),
        'trades_today': 0,
        'is_in_position': False
    }
    
    def _set_headers(self, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
    
    def do_GET(self):
        if self.path == '/health':
            # Actualizar uptime
            start_time = datetime.fromisoformat(self.bot_status['start_time'])
            uptime_seconds = (datetime.now() - start_time).total_seconds()
            self.bot_status['uptime'] = uptime_seconds
            
            # Enviar respuesta
            self._set_headers()
            self.wfile.write(json.dumps(self.bot_status).encode())
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'Not found'}).encode())
    
    @classmethod
    def update_status(cls, status_data):
        """Actualiza el estado del bot para health check"""
        cls.bot_status.update(status_data)
        cls.bot_status['last_update'] = datetime.now().isoformat()


class HealthCheckServer:
    """Servidor HTTP simple para health checks"""
    
    def __init__(self, port=8080):
        self.port = port
        self.server = None
        self.thread = None
        self.running = False
    
    def start(self):
        """Inicia el servidor HTTP en un thread separado"""
        if self.running:
            return
        
        def run_server():
            logger.info(f"Iniciando servidor de health check en puerto {self.port}")
            try:
                self.server = HTTPServer(('0.0.0.0', self.port), HealthCheckHandler)
                self.running = True
                self.server.serve_forever()
            except Exception as e:
                logger.error(f"Error en servidor de health check: {e}")
                self.running = False
        
        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
        
        # Esperar a que el servidor esté listo
        max_wait = 5
        start_time = time.time()
        while not self.running and time.time() - start_time < max_wait:
            time.sleep(0.1)
        
        logger.info("Servidor de health check iniciado correctamente" if self.running else "Error al iniciar servidor de health check")
        return self.running
    
    def stop(self):
        """Detiene el servidor HTTP"""
        if not self.running:
            return
        
        logger.info("Deteniendo servidor de health check...")
        if self.server:
            self.server.shutdown()
            self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
        
        logger.info("Servidor de health check detenido")
    
    @staticmethod
    def update_bot_status(status_data):
        """Actualiza el estado del bot para health check"""
        HealthCheckHandler.update_status(status_data)
        