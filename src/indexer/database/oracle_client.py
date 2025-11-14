"""Cliente para conexão com Oracle Database"""

from typing import Any, List, Dict, Optional
import oracledb
from loguru import logger
from contextlib import contextmanager

from ..config import OracleSettings


class OracleClient:
    """Cliente para interação com Oracle Database"""

    def __init__(self, settings: OracleSettings):
        """
        Inicializa o cliente Oracle

        Args:
            settings: Configurações do Oracle
        """
        self.settings = settings
        self.pool: Optional[oracledb.ConnectionPool] = None
        self._initialize_pool()

    def _initialize_pool(self):
        """Inicializa o connection pool"""
        try:
            self.pool = oracledb.create_pool(
                user=self.settings.user,
                password=self.settings.password,
                dsn=self.settings.dsn,
                min=2,
                max=10,
                increment=1,
                threaded=True
            )
            logger.info(f"Connection pool criado com sucesso para {self.settings.dsn}")
        except Exception as e:
            logger.error(f"Erro ao criar connection pool: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Context manager para obter conexão do pool"""
        connection = None
        try:
            connection = self.pool.acquire()
            yield connection
        except Exception as e:
            logger.error(f"Erro ao adquirir conexão: {e}")
            raise
        finally:
            if connection:
                self.pool.release(connection)

    def fetch_documents(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca documentos do Oracle

        Args:
            limit: Número máximo de documentos
            offset: Offset para paginação
            filters: Filtros adicionais (where clause)

        Returns:
            Lista de documentos
        """
        query = f"SELECT * FROM {self.settings.table}"

        # Adiciona filtros se fornecidos
        if filters:
            where_clauses = [f"{key} = :{key}" for key in filters.keys()]
            query += " WHERE " + " AND ".join(where_clauses)

        # Adiciona ordenação e paginação
        query += " ORDER BY id"

        if limit:
            query += f" OFFSET {offset} ROWS FETCH NEXT {limit} ROWS ONLY"

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                if filters:
                    cursor.execute(query, filters)
                else:
                    cursor.execute(query)

                # Obtém os nomes das colunas
                columns = [col[0].lower() for col in cursor.description]

                # Converte rows para dicionários
                documents = []
                for row in cursor:
                    doc = dict(zip(columns, row))
                    documents.append(doc)

                logger.info(f"Buscados {len(documents)} documentos do Oracle")
                return documents

        except Exception as e:
            logger.error(f"Erro ao buscar documentos: {e}")
            raise

    def fetch_document_by_id(self, doc_id: Any) -> Optional[Dict[str, Any]]:
        """
        Busca um documento específico por ID

        Args:
            doc_id: ID do documento

        Returns:
            Documento ou None se não encontrado
        """
        query = f"SELECT * FROM {self.settings.table} WHERE id = :id"

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, {"id": doc_id})

                row = cursor.fetchone()
                if not row:
                    return None

                columns = [col[0].lower() for col in cursor.description]
                document = dict(zip(columns, row))

                logger.info(f"Documento {doc_id} encontrado")
                return document

        except Exception as e:
            logger.error(f"Erro ao buscar documento {doc_id}: {e}")
            raise

    def update_document_status(
        self,
        doc_id: Any,
        status: str,
        indexed_at: Optional[str] = None
    ) -> bool:
        """
        Atualiza o status de indexação de um documento

        Args:
            doc_id: ID do documento
            status: Novo status
            indexed_at: Data/hora da indexação

        Returns:
            True se atualizado com sucesso
        """
        query = f"UPDATE {self.settings.table} SET status = :status"
        params = {"status": status, "id": doc_id}

        if indexed_at:
            query += ", indexed_at = :indexed_at"
            params["indexed_at"] = indexed_at

        query += " WHERE id = :id"

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()

                logger.info(f"Status do documento {doc_id} atualizado para {status}")
                return True

        except Exception as e:
            logger.error(f"Erro ao atualizar status do documento {doc_id}: {e}")
            raise

    def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Conta o número de documentos

        Args:
            filters: Filtros opcionais

        Returns:
            Número de documentos
        """
        query = f"SELECT COUNT(*) FROM {self.settings.table}"

        if filters:
            where_clauses = [f"{key} = :{key}" for key in filters.keys()]
            query += " WHERE " + " AND ".join(where_clauses)

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                if filters:
                    cursor.execute(query, filters)
                else:
                    cursor.execute(query)

                count = cursor.fetchone()[0]
                return count

        except Exception as e:
            logger.error(f"Erro ao contar documentos: {e}")
            raise

    def close(self):
        """Fecha o connection pool"""
        if self.pool:
            self.pool.close()
            logger.info("Connection pool fechado")
