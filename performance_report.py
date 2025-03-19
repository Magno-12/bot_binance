import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse
import logging
from tabulate import tabulate

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("report_generator")

class PerformanceReportGenerator:
    """Clase para generar informes detallados de rendimiento"""
    
    def __init__(self, data_file='performance_data.json', output_dir='reports'):
        self.data_file = data_file
        self.output_dir = output_dir
        
        # Crear directorio de salida si no existe
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def load_data(self):
        """Carga los datos de rendimiento desde un archivo JSON"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    trades = data.get('trades', [])
                    equity_curve = data.get('equity_curve', [])
                    logger.info(f"Datos cargados: {len(trades)} operaciones")
                    return trades, equity_curve
            else:
                logger.error(f"Archivo de datos no encontrado: {self.data_file}")
                return [], []
        except Exception as e:
            logger.error(f"Error al cargar datos: {e}")
            return [], []
    
    def calculate_statistics(self, trades):
        """Calcula estadísticas detalladas de rendimiento"""
        if not trades:
            logger.warning("No hay operaciones para analizar")
            return {}
        
        # Conversión a DataFrame para análisis más sencillo
        df = pd.DataFrame(trades)
        
        # Estadísticas básicas
        total_trades = len(trades)
        winning_trades = len(df[df['profit_loss'] > 0])
        losing_trades = len(df[df['profit_loss'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L
        total_profit = df[df['profit_loss'] > 0]['profit_loss'].sum()
        total_loss = abs(df[df['profit_loss'] < 0]['profit_loss'].sum())
        net_profit = df['profit_loss'].sum()
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Promedios
        avg_profit = df[df['profit_loss'] > 0]['profit_loss'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['profit_loss'] < 0]['profit_loss'].mean() if losing_trades > 0 else 0
        avg_trade = df['profit_loss'].mean()
        
        # Análisis por side (LONG vs SHORT)
        long_trades = len(df[df['side'] == 'LONG'])
        short_trades = len(df[df['side'] == 'SHORT'])
        
        long_win_rate = (len(df[(df['side'] == 'LONG') & (df['profit_loss'] > 0)]) / long_trades * 100) if long_trades > 0 else 0
        short_win_rate = (len(df[(df['side'] == 'SHORT') & (df['profit_loss'] > 0)]) / short_trades * 100) if short_trades > 0 else 0
        
        long_profit = df[df['side'] == 'LONG']['profit_loss'].sum()
        short_profit = df[df['side'] == 'SHORT']['profit_loss'].sum()
        
        # Calcular drawdown
        if 'profit_loss_percent' in df.columns:
            df['cumulative_return'] = (1 + df['profit_loss_percent'] / 100).cumprod() * 100 - 100
            df['rolling_max'] = df['cumulative_return'].cummax()
            df['drawdown'] = df['rolling_max'] - df['cumulative_return']
            max_drawdown = df['drawdown'].max()
        else:
            max_drawdown = 0
        
        # Análisis por razones de salida
        exit_reasons = df['exit_reason'].value_counts().to_dict() if 'exit_reason' in df.columns else {}
        
        # Cálculo de Sharpe Ratio (simplificado, suponiendo rendimientos anualizados)
        if 'profit_loss_percent' in df.columns:
            returns = df['profit_loss_percent'] / 100
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calcular rendimiento anualizado (estimado)
        if 'entry_time' in df.columns and 'exit_time' in df.columns:
            try:
                first_date = pd.to_datetime(df['entry_time'].min())
                last_date = pd.to_datetime(df['exit_time'].max())
                days = (last_date - first_date).days
                if days > 0:
                    annualized_return = ((1 + net_profit / 100) ** (365 / days) - 1) * 100
                else:
                    annualized_return = 0
            except:
                annualized_return = 0
        else:
            annualized_return = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': net_profit,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'annualized_return': annualized_return,
            'long_trades': long_trades,
            'short_trades': short_trades,
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
            'long_profit': long_profit,
            'short_profit': short_profit,
            'exit_reasons': exit_reasons
        }
    
    def generate_text_report(self, stats, filename=None):
        """Genera un informe de texto detallado"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/performance_report_{timestamp}.txt"
        
        report_lines = [
            "================================================================",
            "               INFORME DE RENDIMIENTO DEL BOT DE TRADING        ",
            "================================================================",
            f"Fecha del informe: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "ESTADÍSTICAS GENERALES",
            "----------------------",
            f"Total de operaciones: {stats['total_trades']}",
            f"Operaciones ganadoras: {stats['winning_trades']} ({stats['win_rate']:.2f}%)",
            f"Operaciones perdedoras: {stats['losing_trades']} ({100-stats['win_rate']:.2f}%)",
            f"Rentabilidad neta: {stats['net_profit']:.6f}",
            f"Rentabilidad anualizada (est.): {stats['annualized_return']:.2f}%",
            "",
            "MÉTRICAS DE RENDIMIENTO",
            "-----------------------",
            f"Ratio de beneficio/pérdida: {stats['profit_factor']:.2f}",
            f"Beneficio promedio: {stats['avg_profit']:.6f}",
            f"Pérdida promedio: {stats['avg_loss']:.6f}",
            f"Operación promedio: {stats['avg_trade']:.6f}",
            f"Drawdown máximo: {stats['max_drawdown']:.2f}%",
            f"Ratio de Sharpe: {stats['sharpe_ratio']:.2f}",
            "",
            "ANÁLISIS POR TIPO DE POSICIÓN",
            "-----------------------------",
            f"Operaciones LONG: {stats['long_trades']} ({stats['long_trades']/stats['total_trades']*100 if stats['total_trades'] > 0 else 0:.2f}%)",
            f"Operaciones SHORT: {stats['short_trades']} ({stats['short_trades']/stats['total_trades']*100 if stats['total_trades'] > 0 else 0:.2f}%)",
            f"Win rate LONG: {stats['long_win_rate']:.2f}%",
            f"Win rate SHORT: {stats['short_win_rate']:.2f}%",
            f"Rentabilidad LONG: {stats['long_profit']:.6f}",
            f"Rentabilidad SHORT: {stats['short_profit']:.6f}",
            "",
            "RAZONES DE SALIDA",
            "----------------"
        ]
        
        # Añadir razones de salida
        for reason, count in stats.get('exit_reasons', {}).items():
            percentage = (count / stats['total_trades']) * 100 if stats['total_trades'] > 0 else 0
            report_lines.append(f"{reason}: {count} ({percentage:.2f}%)")
        
        # Escribir el informe a un archivo
        with open(filename, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Informe de texto generado: {filename}")
        return filename
    
    def generate_html_report(self, stats, trades, equity_curve, filename=None):
        """Genera un informe HTML con gráficos"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/performance_report_{timestamp}.html"
        
        # Convertir trades a DataFrame para procesamiento
        df_trades = pd.DataFrame(trades)
        if not df_trades.empty and 'entry_time' in df_trades.columns:
            df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
            if 'exit_time' in df_trades.columns:
                df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
        
        # Crear DataFrame para la curva de equidad
        df_equity = pd.DataFrame(equity_curve)
        if not df_equity.empty and 'timestamp' in df_equity.columns:
            df_equity['timestamp'] = pd.to_datetime(df_equity['timestamp'])
        
        # Generar gráficos y guardarlos como archivos
        charts_paths = self._generate_charts(df_trades, df_equity)
        
        # Crear tabla HTML para estadísticas
        stats_table = f"""
        <table class="stats-table">
            <tr>
                <th colspan="2">Estadísticas Generales</th>
            </tr>
            <tr>
                <td>Total de operaciones</td>
                <td>{stats['total_trades']}</td>
            </tr>
            <tr>
                <td>Win Rate</td>
                <td>{stats['win_rate']:.2f}%</td>
            </tr>
            <tr>
                <td>Profit Factor</td>
                <td>{stats['profit_factor']:.2f}</td>
            </tr>
            <tr>
                <td>Rentabilidad Neta</td>
                <td>{stats['net_profit']:.6f}</td>
            </tr>
            <tr>
                <td>Drawdown Máximo</td>
                <td>{stats['max_drawdown']:.2f}%</td>
            </tr>
            <tr>
                <td>Ratio de Sharpe</td>
                <td>{stats['sharpe_ratio']:.2f}</td>
            </tr>
            <tr>
                <td>Rentabilidad Anualizada</td>
                <td>{stats['annualized_return']:.2f}%</td>
            </tr>
        </table>
        """
        
        # Crear tabla de operaciones recientes
        recent_trades = df_trades.tail(10).to_dict('records') if not df_trades.empty else []
        trades_table = """
        <table class="trades-table">
            <tr>
                <th>Fecha Entrada</th>
                <th>Lado</th>
                <th>Precio Entrada</th>
                <th>Precio Salida</th>
                <th>P&L %</th>
                <th>Razón Salida</th>
            </tr>
        """
        
        for trade in reversed(recent_trades):
            entry_time = trade.get('entry_time', '')
            if isinstance(entry_time, pd.Timestamp):
                entry_time = entry_time.strftime('%Y-%m-%d %H:%M')
            
            exit_time = trade.get('exit_time', '')
            if isinstance(exit_time, pd.Timestamp):
                exit_time = exit_time.strftime('%Y-%m-%d %H:%M')
                
            trades_table += f"""
            <tr>
                <td>{entry_time}</td>
                <td>{trade.get('side', '')}</td>
                <td>{trade.get('entry_price', 0):.2f}</td>
                <td>{trade.get('exit_price', 0):.2f}</td>
                <td>{trade.get('profit_loss_percent', 0):.2f}%</td>
                <td>{trade.get('exit_reason', '')}</td>
            </tr>
            """
        
        trades_table += "</table>"
        
        # Crear el HTML completo
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Informe de Rendimiento del Bot de Trading</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    color: #333;
                }}
                h1, h2 {{
                    color: #2c3e50;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .stats-container {{
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 30px;
                }}
                .chart-container {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .stats-table, .trades-table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                .stats-table th, .stats-table td, .trades-table th, .trades-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .stats-table th, .trades-table th {{
                    background-color: #f2f2f2;
                    color: #333;
                }}
                .trades-table tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .positive {{
                    color: green;
                }}
                .negative {{
                    color: red;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    margin-bottom: 15px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Informe de Rendimiento del Bot de Trading</h1>
                <p>Fecha del informe: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="stats-container">
                    {stats_table}
                </div>
                
                <h2>Gráficos de Rendimiento</h2>
                <div class="chart-container">
                    <img src="{charts_paths.get('equity_curve', '')}" alt="Curva de Equidad">
                    <img src="{charts_paths.get('drawdown', '')}" alt="Drawdown">
                    <img src="{charts_paths.get('win_loss', '')}" alt="Distribución de Ganancias/Pérdidas">
                    <img src="{charts_paths.get('trade_duration', '')}" alt="Duración de Operaciones">
                </div>
                
                <h2>Operaciones Recientes</h2>
                {trades_table}
            </div>
        </body>
        </html>
        """
        
        # Escribir el HTML a un archivo
        with open(filename, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Informe HTML generado: {filename}")
        return filename
    
    def _generate_charts(self, df_trades, df_equity):
        """Genera gráficos para el informe y devuelve las rutas"""
        chart_paths = {}
        
        # Crear directorio para gráficos si no existe
        charts_dir = f"{self.output_dir}/charts"
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Curva de equidad
        if not df_equity.empty and 'equity' in df_equity.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df_equity['timestamp'], df_equity['equity'], 'b-')
            plt.title('Curva de Equidad')
            plt.xlabel('Fecha')
            plt.ylabel('Equidad (%)')
            plt.grid(True)
            equity_path = f"{charts_dir}/equity_{timestamp}.png"
            plt.savefig(equity_path)
            plt.close()
            chart_paths['equity_curve'] = equity_path
        
        # 2. Drawdown
        if not df_equity.empty and 'equity' in df_equity.columns:
            # Calcular drawdown
            df_equity['peak'] = df_equity['equity'].cummax()
            df_equity['drawdown'] = (df_equity['peak'] - df_equity['equity']) / df_equity['peak'] * 100
            
            plt.figure(figsize=(10, 6))
            plt.plot(df_equity['timestamp'], df_equity['drawdown'], 'r-')
            plt.title('Drawdown')
            plt.xlabel('Fecha')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
            plt.gca().invert_yaxis()  # Invertir eje Y para que drawdown baje
            drawdown_path = f"{charts_dir}/drawdown_{timestamp}.png"
            plt.savefig(drawdown_path)
            plt.close()
            chart_paths['drawdown'] = drawdown_path
        
        # 3. Distribución de ganancias/pérdidas
        if not df_trades.empty and 'profit_loss_percent' in df_trades.columns:
            plt.figure(figsize=(10, 6))
            df_trades['profit_loss_percent'].hist(bins=20, color='skyblue', edgecolor='black')
            plt.title('Distribución de Ganancias/Pérdidas')
            plt.xlabel('Rentabilidad (%)')
            plt.ylabel('Frecuencia')
            plt.grid(True)
            win_loss_path = f"{charts_dir}/win_loss_{timestamp}.png"
            plt.savefig(win_loss_path)
            plt.close()
            chart_paths['win_loss'] = win_loss_path
        
        # 4. Duración de operaciones
        if not df_trades.empty and 'entry_time' in df_trades.columns and 'exit_time' in df_trades.columns:
            try:
                # Calcular duración en horas
                df_trades['duration_hours'] = (df_trades['exit_time'] - df_trades['entry_time']).dt.total_seconds() / 3600
                
                plt.figure(figsize=(10, 6))
                df_trades['duration_hours'].hist(bins=20, color='lightgreen', edgecolor='black')
                plt.title('Duración de Operaciones')
                plt.xlabel('Duración (horas)')
                plt.ylabel('Frecuencia')
                plt.grid(True)
                duration_path = f"{charts_dir}/duration_{timestamp}.png"
                plt.savefig(duration_path)
                plt.close()
                chart_paths['trade_duration'] = duration_path
            except Exception as e:
                logger.error(f"Error al generar gráfico de duración: {e}")
        
        return chart_paths
    
    def generate_report(self, format='text'):
        """Genera un informe completo en el formato especificado"""
        trades, equity_curve = self.load_data()
        
        if not trades:
            logger.warning("No hay datos para generar el informe")
            return None
        
        stats = self.calculate_statistics(trades)
        
        if format.lower() == 'text':
            return self.generate_text_report(stats)
        elif format.lower() == 'html':
            return self.generate_html_report(stats, trades, equity_curve)
        else:
            logger.error(f"Formato de informe no soportado: {format}")
            return None


def main():
    """Función principal para ejecutar el generador de informes desde línea de comandos"""
    parser = argparse.ArgumentParser(description='Generador de informes de rendimiento para el bot de trading')
    parser.add_argument('--data', default='performance_data.json', help='Archivo JSON con datos de rendimiento')
    parser.add_argument('--output', default='reports', help='Directorio de salida para informes')
    parser.add_argument('--format', default='text', choices=['text', 'html'], help='Formato del informe')

    args = parser.parse_args()

    generator = PerformanceReportGenerator(
        data_file=args.data,
        output_dir=args.output
    )

    report_path = generator.generate_report(format=args.format)

    if report_path:
        print(f"Informe generado exitosamente: {report_path}")
    else:
        print("No se pudo generar el informe")


if __name__ == "__main__":
    main()
