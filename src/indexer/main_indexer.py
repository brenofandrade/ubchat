"""
Módulo principal de indexação com contexto enriquecido por LLM

Este módulo orquestra todo o pipeline de indexação:
1. Busca documentos do Oracle
2. Divide em chunks inteligentes
3. Gera contexto com LLM para cada chunk
4. Cria embeddings enriquecidos
5. Armazena no Pinecone com metadata rica
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger
from tqdm import tqdm

from .config import get_settings, Settings
from .database.oracle_client import OracleClient
from .vectorstore.pinecone_client import PineconeClient
from .chunking.text_chunker import TextChunker, ChunkStrategy
from .context.context_generator import ContextGenerator
from .embeddings.embedding_generator import EmbeddingGenerator
from .utils.logger_config import setup_logger


class DocumentIndexer:
    """
    Indexador principal de documentos com enriquecimento por LLM

    Este é o componente central que implementa a estratégia melhorada de indexação,
    onde cada chunk é analisado por LLM para adicionar contexto semântico rico,
    melhorando significativamente a qualidade da recuperação de informação.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
        use_llm_context: bool = True,
        llm_provider: str = "openai"
    ):
        """
        Inicializa o indexador

        Args:
            settings: Configurações do sistema
            chunk_strategy: Estratégia de chunking
            use_llm_context: Se True, enriquece chunks com LLM
            llm_provider: Provider LLM ("openai" ou "anthropic")
        """
        # Carrega configurações
        self.settings = settings or get_settings()

        # Setup logger
        setup_logger(self.settings.logging)

        logger.info("=" * 80)
        logger.info("Inicializando DocumentIndexer com contexto enriquecido por LLM")
        logger.info("=" * 80)

        # Inicializa componentes
        self.oracle_client = OracleClient(self.settings.oracle)
        self.pinecone_client = PineconeClient(
            self.settings.pinecone,
            dimension=3072  # text-embedding-3-large
        )

        self.text_chunker = TextChunker(
            self.settings.chunking,
            strategy=chunk_strategy
        )

        self.use_llm_context = use_llm_context
        if use_llm_context:
            self.context_generator = ContextGenerator(
                openai_settings=self.settings.openai,
                anthropic_settings=self.settings.anthropic,
                context_settings=self.settings.context,
                use_provider=llm_provider
            )

        self.embedding_generator = EmbeddingGenerator(self.settings.openai)

        logger.info(f"Componentes inicializados:")
        logger.info(f"  - Oracle: {self.settings.oracle.dsn}")
        logger.info(f"  - Pinecone: {self.settings.pinecone.index_name}")
        logger.info(f"  - Chunk Strategy: {chunk_strategy.value}")
        logger.info(f"  - LLM Context: {use_llm_context}")
        logger.info(f"  - LLM Provider: {llm_provider}")

    def index_document(
        self,
        doc_id: Any,
        text_field: str = "content",
        namespace: str = "",
        update_oracle_status: bool = True
    ) -> Dict[str, Any]:
        """
        Indexa um documento específico

        Args:
            doc_id: ID do documento no Oracle
            text_field: Nome do campo com o texto
            namespace: Namespace no Pinecone
            update_oracle_status: Se True, atualiza status no Oracle

        Returns:
            Estatísticas da indexação
        """
        logger.info(f"Iniciando indexação do documento {doc_id}")

        try:
            # 1. Busca documento do Oracle
            document = self.oracle_client.fetch_document_by_id(doc_id)
            if not document:
                raise ValueError(f"Documento {doc_id} não encontrado")

            text = document.get(text_field, "")
            if not text:
                raise ValueError(f"Campo '{text_field}' vazio ou não encontrado")

            logger.info(f"Documento carregado: {len(text)} caracteres")

            # 2. Gera contexto geral do documento (se usando LLM)
            doc_context = None
            if self.use_llm_context:
                doc_context = self.context_generator.generate_document_summary(text)
                logger.info(f"Contexto do documento gerado: {doc_context[:100]}...")

            # 3. Divide em chunks
            metadata = {k: v for k, v in document.items() if k != text_field}
            chunks = self.text_chunker.chunk_document(
                text,
                doc_id=str(doc_id),
                metadata=metadata
            )
            logger.info(f"Documento dividido em {len(chunks)} chunks")

            # 4. Gera contexto para cada chunk (se habilitado)
            if self.use_llm_context:
                enriched_chunks = self.context_generator.generate_contexts_batch(
                    chunks,
                    doc_context=doc_context,
                    show_progress=True
                )
                logger.info(f"Contextos gerados para {len(enriched_chunks)} chunks")
            else:
                # Se não usar LLM, cria EnrichedChunks básicos
                from .context.context_generator import EnrichedChunk
                enriched_chunks = [
                    EnrichedChunk(
                        original_chunk=chunk,
                        contextual_summary="",
                        key_concepts=[],
                        keywords=[],
                        topic="",
                        questions=[],
                        enhanced_text=chunk.text
                    )
                    for chunk in chunks
                ]

            # 5. Gera embeddings e cria vetores
            vectors = self.embedding_generator.create_vectors_batch(
                enriched_chunks,
                use_enhanced_text=self.use_llm_context,
                show_progress=True
            )
            logger.info(f"Vetores criados: {len(vectors)}")

            # 6. Insere no Pinecone
            total_upserted = self.pinecone_client.upsert_batch(
                vectors,
                batch_size=100,
                namespace=namespace
            )

            # 7. Atualiza status no Oracle
            if update_oracle_status:
                self.oracle_client.update_document_status(
                    doc_id,
                    status="indexed",
                    indexed_at=datetime.now().isoformat()
                )

            stats = {
                "doc_id": doc_id,
                "chunks": len(chunks),
                "vectors_upserted": total_upserted,
                "total_tokens": sum(c.token_count for c in chunks),
                "use_llm_context": self.use_llm_context
            }

            logger.info(f"Documento {doc_id} indexado com sucesso: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Erro ao indexar documento {doc_id}: {e}")
            if update_oracle_status:
                self.oracle_client.update_document_status(
                    doc_id,
                    status="error"
                )
            raise

    def index_all_documents(
        self,
        text_field: str = "content",
        namespace: str = "",
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        update_oracle_status: bool = True
    ) -> Dict[str, Any]:
        """
        Indexa todos os documentos do Oracle

        Args:
            text_field: Nome do campo com o texto
            namespace: Namespace no Pinecone
            limit: Limite de documentos (None = todos)
            filters: Filtros para buscar documentos
            update_oracle_status: Atualizar status no Oracle

        Returns:
            Estatísticas gerais da indexação
        """
        logger.info("=" * 80)
        logger.info("Iniciando indexação em lote")
        logger.info("=" * 80)

        # Conta total de documentos
        total_docs = self.oracle_client.count_documents(filters)
        docs_to_process = min(total_docs, limit) if limit else total_docs

        logger.info(f"Total de documentos a processar: {docs_to_process}")

        # Busca documentos
        documents = self.oracle_client.fetch_documents(
            limit=limit,
            filters=filters
        )

        # Estatísticas
        stats = {
            "total_documents": len(documents),
            "successful": 0,
            "failed": 0,
            "total_chunks": 0,
            "total_vectors": 0,
            "errors": []
        }

        # Processa cada documento
        for doc in tqdm(documents, desc="Indexando documentos"):
            try:
                doc_id = doc.get("id")
                result = self.index_document(
                    doc_id,
                    text_field=text_field,
                    namespace=namespace,
                    update_oracle_status=update_oracle_status
                )

                stats["successful"] += 1
                stats["total_chunks"] += result["chunks"]
                stats["total_vectors"] += result["vectors_upserted"]

            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append({
                    "doc_id": doc.get("id"),
                    "error": str(e)
                })
                logger.error(f"Falha ao processar documento {doc.get('id')}: {e}")

        logger.info("=" * 80)
        logger.info("Indexação em lote concluída")
        logger.info(f"Sucesso: {stats['successful']}/{stats['total_documents']}")
        logger.info(f"Falhas: {stats['failed']}")
        logger.info(f"Total de chunks: {stats['total_chunks']}")
        logger.info(f"Total de vetores: {stats['total_vectors']}")
        logger.info("=" * 80)

        return stats

    def search(
        self,
        query: str,
        top_k: int = 10,
        namespace: str = "",
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca documentos similares

        Args:
            query: Query de busca
            top_k: Número de resultados
            namespace: Namespace
            filters: Filtros de metadata

        Returns:
            Lista de resultados
        """
        logger.info(f"Buscando: '{query}' (top_k={top_k})")

        # Gera embedding da query
        query_embedding = self.embedding_generator.generate_query_embedding(query)

        # Busca no Pinecone
        response = self.pinecone_client.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filters,
            namespace=namespace
        )

        # Formata resultados
        results = []
        for match in response.matches:
            result = {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata
            }
            results.append(result)

        logger.info(f"Encontrados {len(results)} resultados")
        return results

    def delete_document(
        self,
        doc_id: str,
        namespace: str = ""
    ) -> int:
        """
        Remove todos os chunks de um documento do Pinecone

        Args:
            doc_id: ID do documento
            namespace: Namespace

        Returns:
            Número de vetores removidos
        """
        logger.info(f"Removendo documento {doc_id} do índice")

        self.pinecone_client.delete_by_filter(
            filter={"doc_id": doc_id},
            namespace=namespace
        )

        logger.info(f"Documento {doc_id} removido")
        return 1

    def get_stats(self, namespace: str = "") -> Dict[str, Any]:
        """
        Obtém estatísticas do índice

        Args:
            namespace: Namespace

        Returns:
            Estatísticas
        """
        return self.pinecone_client.get_index_stats(namespace)

    def close(self):
        """Fecha conexões"""
        self.oracle_client.close()
        logger.info("Indexador encerrado")
