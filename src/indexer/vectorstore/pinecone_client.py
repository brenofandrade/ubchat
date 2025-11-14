"""Cliente para conexão com Pinecone"""

from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from loguru import logger
import time

from ..config import PineconeSettings


class PineconeClient:
    """Cliente para interação com Pinecone vector database"""

    def __init__(self, settings: PineconeSettings, dimension: int = 3072):
        """
        Inicializa o cliente Pinecone

        Args:
            settings: Configurações do Pinecone
            dimension: Dimensão dos vetores (3072 para text-embedding-3-large)
        """
        self.settings = settings
        self.dimension = dimension
        self.pc = Pinecone(api_key=settings.api_key)
        self.index = None
        self._initialize_index()

    def _initialize_index(self):
        """Inicializa ou cria o índice Pinecone"""
        try:
            # Verifica se o índice existe
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]

            if self.settings.index_name not in index_names:
                logger.info(f"Criando novo índice: {self.settings.index_name}")

                # Cria o índice com serverless spec
                self.pc.create_index(
                    name=self.settings.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.settings.environment
                    )
                )

                # Aguarda o índice ficar pronto
                while not self.pc.describe_index(self.settings.index_name).status['ready']:
                    logger.info("Aguardando índice ficar pronto...")
                    time.sleep(1)

                logger.info(f"Índice {self.settings.index_name} criado com sucesso")
            else:
                logger.info(f"Usando índice existente: {self.settings.index_name}")

            # Conecta ao índice
            self.index = self.pc.Index(self.settings.index_name)

        except Exception as e:
            logger.error(f"Erro ao inicializar índice Pinecone: {e}")
            raise

    def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = ""
    ) -> Dict[str, Any]:
        """
        Insere ou atualiza vetores no Pinecone

        Args:
            vectors: Lista de vetores com formato:
                     [{"id": "doc1_chunk1", "values": [...], "metadata": {...}}]
            namespace: Namespace opcional para organização

        Returns:
            Resposta do Pinecone com contagem de vetores inseridos
        """
        try:
            response = self.index.upsert(
                vectors=vectors,
                namespace=namespace
            )

            logger.info(
                f"Inseridos {response.upserted_count} vetores no namespace '{namespace}'"
            )
            return response

        except Exception as e:
            logger.error(f"Erro ao inserir vetores: {e}")
            raise

    def upsert_batch(
        self,
        vectors: List[Dict[str, Any]],
        batch_size: int = 100,
        namespace: str = ""
    ) -> int:
        """
        Insere vetores em batches para melhor performance

        Args:
            vectors: Lista de vetores
            batch_size: Tamanho do batch
            namespace: Namespace opcional

        Returns:
            Total de vetores inseridos
        """
        total_upserted = 0

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            response = self.upsert_vectors(batch, namespace)
            total_upserted += response.upserted_count

            logger.info(
                f"Batch {i // batch_size + 1}: "
                f"{response.upserted_count} vetores inseridos"
            )

        logger.info(f"Total de {total_upserted} vetores inseridos com sucesso")
        return total_upserted

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        namespace: str = "",
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Consulta vetores similares

        Args:
            vector: Vetor de consulta
            top_k: Número de resultados
            filter: Filtros de metadata
            namespace: Namespace para busca
            include_metadata: Incluir metadata nos resultados

        Returns:
            Resultados da consulta
        """
        try:
            response = self.index.query(
                vector=vector,
                top_k=top_k,
                filter=filter,
                namespace=namespace,
                include_metadata=include_metadata
            )

            logger.info(f"Query retornou {len(response.matches)} resultados")
            return response

        except Exception as e:
            logger.error(f"Erro ao consultar vetores: {e}")
            raise

    def delete_by_ids(
        self,
        ids: List[str],
        namespace: str = ""
    ) -> Dict[str, Any]:
        """
        Remove vetores por IDs

        Args:
            ids: Lista de IDs para remover
            namespace: Namespace

        Returns:
            Resposta do Pinecone
        """
        try:
            response = self.index.delete(
                ids=ids,
                namespace=namespace
            )

            logger.info(f"Removidos {len(ids)} vetores do namespace '{namespace}'")
            return response

        except Exception as e:
            logger.error(f"Erro ao remover vetores: {e}")
            raise

    def delete_by_filter(
        self,
        filter: Dict[str, Any],
        namespace: str = ""
    ) -> Dict[str, Any]:
        """
        Remove vetores por filtro de metadata

        Args:
            filter: Filtro de metadata
            namespace: Namespace

        Returns:
            Resposta do Pinecone
        """
        try:
            response = self.index.delete(
                filter=filter,
                namespace=namespace
            )

            logger.info(f"Vetores removidos com filtro: {filter}")
            return response

        except Exception as e:
            logger.error(f"Erro ao remover vetores por filtro: {e}")
            raise

    def get_index_stats(self, namespace: str = "") -> Dict[str, Any]:
        """
        Obtém estatísticas do índice

        Args:
            namespace: Namespace opcional

        Returns:
            Estatísticas do índice
        """
        try:
            stats = self.index.describe_index_stats()

            if namespace:
                namespace_stats = stats.namespaces.get(namespace, {})
                logger.info(f"Stats do namespace '{namespace}': {namespace_stats}")
                return namespace_stats
            else:
                logger.info(f"Stats do índice: {stats}")
                return stats

        except Exception as e:
            logger.error(f"Erro ao obter estatísticas: {e}")
            raise

    def fetch_vectors(
        self,
        ids: List[str],
        namespace: str = ""
    ) -> Dict[str, Any]:
        """
        Busca vetores específicos por ID

        Args:
            ids: Lista de IDs
            namespace: Namespace

        Returns:
            Vetores encontrados
        """
        try:
            response = self.index.fetch(
                ids=ids,
                namespace=namespace
            )

            logger.info(f"Buscados {len(response.vectors)} vetores")
            return response

        except Exception as e:
            logger.error(f"Erro ao buscar vetores: {e}")
            raise
