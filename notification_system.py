# notification_system.py
import smtplib
import ssl
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

logger = logging.getLogger("notification_system")

class NotificationSystem:
    """Sistema para enviar notificaciones sobre eventos importantes del bot"""
    
    def __init__(self):
        # Cargar configuración desde variables de entorno
        self.email_enabled = os.environ.get('ENABLE_EMAIL_NOTIFICATIONS', 'False').lower() == 'true'
        self.email_sender = os.environ.get('EMAIL_SENDER', '')
        self.email_password = os.environ.get('EMAIL_PASSWORD', '')
        self.email_recipient = os.environ.get('EMAIL_RECIPIENT', '')
        self.email_server = os.environ.get('EMAIL_SERVER', 'smtp.gmail.com')
        self.email_port = int(os.environ.get('EMAIL_PORT', '587'))
        
        # Verificar configuración
        if self.email_enabled:
            if not all([self.email_sender, self.email_password, self.email_recipient]):
                logger.warning("Configuración de correo electrónico incompleta. Las notificaciones no funcionarán.")
                self.email_enabled = False
            else:
                logger.info("Sistema de notificaciones por correo electrónico inicializado")
        else:
            logger.info("Notificaciones por correo electrónico desactivadas")
    
    def send_email(self, subject, body):
        """Envía un correo electrónico de notificación"""
        if not self.email_enabled:
            logger.debug(f"Notificación no enviada (desactivado): {subject}")
            return False
        
        try:
            # Crear mensaje
            message = MIMEMultipart()
            message["From"] = self.email_sender
            message["To"] = self.email_recipient
            message["Subject"] = f"[BOT TRADING] {subject}"
            
            # Añadir cuerpo del mensaje
            message.attach(MIMEText(body, "plain"))
            
            # Crear conexión segura y enviar
            context = ssl.create_default_context()
            with smtplib.SMTP(self.email_server, self.email_port) as server:
                server.starttls(context=context)
                server.login(self.email_sender, self.email_password)
                server.sendmail(self.email_sender, self.email_recipient, message.as_string())
            
            logger.info(f"Notificación enviada: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Error al enviar notificación por correo: {e}")
            return False
    
    def notify_trade_executed(self, trade_data):
        """Notifica sobre una operación ejecutada"""
        subject = f"Nueva operación: {trade_data['side']} {trade_data['symbol']}"
        body = f"""
Se ha ejecutado una nueva operación:

Símbolo: {trade_data['symbol']}
Tipo: {trade_data['side']}
Precio de entrada: {trade_data['entry_price']}
Cantidad: {trade_data['quantity']}
Stop Loss: {trade_data.get('stop_loss', 'No configurado')}
Take Profit: {trade_data.get('take_profit', 'No configurado')}
Fecha/Hora: {trade_data.get('entry_time', 'No disponible')}

Este es un mensaje automático del bot de trading.
        """
        return self.send_email(subject, body)
    
    def notify_trade_closed(self, trade_data):
        """Notifica sobre el cierre de una operación"""
        subject = f"Operación cerrada: {trade_data['side']} {trade_data['symbol']} ({trade_data.get('profit_loss_percent', 0):.2f}%)"
        body = f"""
Se ha cerrado una operación:

Símbolo: {trade_data['symbol']}
Tipo: {trade_data['side']}
Precio de entrada: {trade_data['entry_price']}
Precio de salida: {trade_data['exit_price']}
Ganancia/Pérdida: {trade_data.get('profit_loss_percent', 0):.2f}%
Ganancia/Pérdida absoluta: {trade_data.get('profit_loss', 0):.6f}
Razón de salida: {trade_data.get('exit_reason', 'No especificada')}
Duración: {trade_data.get('duration', 'No disponible')}

Este es un mensaje automático del bot de trading.
        """
        return self.send_email(subject, body)
    
    def notify_error(self, error_message, error_details=None):
        """Notifica sobre un error crítico"""
        subject = f"ERROR CRÍTICO en el bot de trading"
        body = f"""
Se ha producido un error crítico en el bot de trading:

ERROR: {error_message}

{error_details if error_details else ''}

Por favor, revise los logs y el estado del bot.
Este es un mensaje automático del sistema de monitoreo.
        """
        return self.send_email(subject, body)
    
    def notify_restart(self, reason=None):
        """Notifica sobre un reinicio del bot"""
        subject = f"El bot de trading se ha reiniciado"
        body = f"""
El bot de trading se ha reiniciado.

Razón: {reason if reason else 'No especificada'}

El sistema continuará operando normalmente.
Este es un mensaje automático del sistema de monitoreo.
        """
        return self.send_email(subject, body)
    
    def notify_daily_summary(self, stats):
        """Envía un resumen diario de actividad"""
        subject = f"Resumen diario del bot de trading"
        
        # Crear cuerpo del mensaje con estadísticas
        body = """
RESUMEN DIARIO DE ACTIVIDAD DEL BOT DE TRADING

Operaciones hoy:
- Total: {total_trades}
- Ganadoras: {winning_trades} ({win_rate:.2f}%)
- Perdedoras: {losing_trades}

Rendimiento:
- P&L total del día: {total_pnl:.6f}
- Mejor operación: {best_trade:.2f}%
- Peor operación: {worst_trade:.2f}%

Balance actual: {current_balance:.2f} USDT

Este es un informe automático diario.
        """.format(**stats)
        
        return self.send_email(subject, body)