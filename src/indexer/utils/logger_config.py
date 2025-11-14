"""Configuração de logging"""

import sys
from loguru import logger
from ..config import LoggingSettings


def setup_logger(settings: LoggingSettings):
    """
    Configura o logger da aplicação

    Args:
        settings: Configurações de logging
    """
    # Remove handlers padrão
    logger.remove()

    # Adiciona handler para console
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True
    )

    # Adiciona handler para arquivo
    logger.add(
        settings.log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.log_level,
        rotation="10 MB",
        retention="7 days",
        compression="zip"
    )

    logger.info("Logger configurado com sucesso")
