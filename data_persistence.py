# data_persistence.py
import os
import json
import shutil
import time
from datetime import datetime, timedelta
import logging
import threading

logger = logging.getLogger("data_persistence")

class DataPersistence:
    """Gestiona la persistencia y copias de seguridad de datos críticos"""
    
    def __init__(self, config=None):
        if config is None:
            config = {}
        
        # Configuración de rutas
        self.data_dir = config.get('data_dir', 'data')
        self.backup_dir = config.get('backup_dir', 'backups')
        
        # Configuración de intervalos
        self.backup_interval = config.get('backup_interval', 3600)  # En segundos (1 hora por defecto)
        self.max_backups = config.get('max_backups', 24)  # Mantener máximo 24 backups
        
        # Crear directorios si no existen
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Iniciar hilo de backup automático
        self._stop_backup_thread = threading.Event()
        self._backup_thread = threading.Thread(target=self._automatic_backup_thread, daemon=True)
        self._backup_thread.start()
        
        logger.info(f"Sistema de persistencia de datos inicializado. Dir datos: {self.data_dir}, Dir backups: {self.backup_dir}")
    
    def _automatic_backup_thread(self):
        """Thread para realizar backups automáticos"""
        while not self._stop_backup_thread.is_set():
            try:
                # Dormir primero para evitar backup inmediato al inicio
                for _ in range(self.backup_interval):
                    if self._stop_backup_thread.is_set():
                        break
                    time.sleep(1)
                
                if not self._stop_backup_thread.is_set():
                    self.create_backup()
                    self.cleanup_old_backups()
                    
            except Exception as e:
                logger.error(f"Error en el hilo de backup automático: {e}")
                time.sleep(60)  # Esperar un minuto antes de reintentar
    
    def stop(self):
        """Detiene el hilo de backup automático"""
        self._stop_backup_thread.set()
        if self._backup_thread.is_alive():
            self._backup_thread.join(timeout=5)
        logger.info("Hilo de backup automático detenido")
    
    def save_data(self, data, filename):
        """Guarda datos en el directorio de datos"""
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            # Crear copia temporal primero
            temp_filepath = f"{filepath}.tmp"
            with open(temp_filepath, 'w') as f:
                if isinstance(data, str):
                    f.write(data)
                else:
                    json.dump(data, f, indent=4)
            
            # Renombrar al archivo final (operación atómica)
            os.replace(temp_filepath, filepath)
            logger.debug(f"Datos guardados correctamente en {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error al guardar datos en {filepath}: {e}")
            return False
    
    def load_data(self, filename, default=None):
        """Carga datos desde el directorio de datos"""
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Archivo {filepath} no encontrado")
                return default
            
            with open(filepath, 'r') as f:
                if filename.endswith('.json'):
                    return json.load(f)
                else:
                    return f.read()
                
        except Exception as e:
            logger.error(f"Error al cargar datos desde {filepath}: {e}")
            # Intentar recuperar desde backup si existe
            backup_data = self.restore_from_latest_backup(filename)
            if backup_data is not None:
                logger.info(f"Datos recuperados desde backup para {filename}")
                return backup_data
            return default
    
    def create_backup(self):
        """Crea una copia de seguridad de los datos actuales"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"backup_{timestamp}")
        
        try:
            os.makedirs(backup_path, exist_ok=True)
            
            # Copiar todos los archivos del directorio de datos
            for filename in os.listdir(self.data_dir):
                source_path = os.path.join(self.data_dir, filename)
                if os.path.isfile(source_path):
                    dest_path = os.path.join(backup_path, filename)
                    shutil.copy2(source_path, dest_path)
            
            logger.info(f"Backup creado correctamente en {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Error al crear backup: {e}")
            return None
    
    def cleanup_old_backups(self):
        """Elimina backups antiguos excediendo el límite máximo"""
        try:
            # Listar todos los directorios de backup
            backup_dirs = []
            for dirname in os.listdir(self.backup_dir):
                if dirname.startswith("backup_"):
                    dirpath = os.path.join(self.backup_dir, dirname)
                    if os.path.isdir(dirpath):
                        backup_dirs.append((dirname, os.path.getctime(dirpath)))
            
            # Ordenar por fecha de creación (más reciente primero)
            backup_dirs.sort(key=lambda x: x[1], reverse=True)
            
            # Eliminar backups excedentes
            for dirname, _ in backup_dirs[self.max_backups:]:
                dirpath = os.path.join(self.backup_dir, dirname)
                shutil.rmtree(dirpath)
                logger.info(f"Backup antiguo eliminado: {dirpath}")
                
        except Exception as e:
            logger.error(f"Error al limpiar backups antiguos: {e}")
    
    def restore_from_backup(self, backup_path):
        """Restaura datos desde un backup específico"""
        try:
            if not os.path.exists(backup_path) or not os.path.isdir(backup_path):
                logger.error(f"Ruta de backup no válida: {backup_path}")
                return False
            
            # Copiar archivos del backup al directorio de datos
            for filename in os.listdir(backup_path):
                source_path = os.path.join(backup_path, filename)
                if os.path.isfile(source_path):
                    dest_path = os.path.join(self.data_dir, filename)
                    shutil.copy2(source_path, dest_path)
            
            logger.info(f"Datos restaurados desde backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error al restaurar desde backup: {e}")
            return False
    
    def restore_from_latest_backup(self, filename=None):
        """Restaura un archivo específico desde el backup más reciente"""
        try:
            # Encontrar el backup más reciente
            backup_dirs = []
            for dirname in os.listdir(self.backup_dir):
                if dirname.startswith("backup_"):
                    dirpath = os.path.join(self.backup_dir, dirname)
                    if os.path.isdir(dirpath):
                        backup_dirs.append((dirname, os.path.getctime(dirpath)))
            
            if not backup_dirs:
                logger.warning("No se encontraron backups disponibles")
                return None
            
            # Ordenar por fecha de creación (más reciente primero)
            backup_dirs.sort(key=lambda x: x[1], reverse=True)
            latest_backup = os.path.join(self.backup_dir, backup_dirs[0][0])
            
            # Si se especifica un archivo, restaurar solo ese archivo
            if filename:
                source_path = os.path.join(latest_backup, filename)
                if os.path.exists(source_path) and os.path.isfile(source_path):
                    with open(source_path, 'r') as f:
                        if filename.endswith('.json'):
                            return json.load(f)
                        else:
                            return f.read()
                else:
                    logger.warning(f"Archivo {filename} no encontrado en el backup más reciente")
                    return None
            
            # Si no se especifica archivo, restaurar todo el backup
            else:
                return self.restore_from_backup(latest_backup)
            
        except Exception as e:
            logger.error(f"Error al restaurar desde el backup más reciente: {e}")
            return None
