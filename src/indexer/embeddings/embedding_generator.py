"""Gerador de embeddings para chunks enriquecidos"""

from typing import List, Dict, Any
import openai
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from ..config import OpenAISettings
from ..context.context_generator import EnrichedChunk


class EmbeddingGenerator:
    """
    Gera embeddings para chunks enriquecidos

    Esta classe utiliza o texto enriquecido (com contexto LLM) para gerar
    embeddings mais informativos, melhorando a recuperação semântica
    """

    def __init__(self, settings: OpenAISettings):
        """
        Inicializa o gerador de embeddings

        Args:
            settings: Configurações OpenAI
        """
        self.settings = settings
        openai.api_key = settings.api_key
        self.model = settings.embedding_model
        self.dimension = self._get_embedding_dimension()

        logger.info(f"EmbeddingGenerator inicializado com modelo {self.model}")

    def _get_embedding_dimension(self) -> int:
        """Retorna a dimensão do modelo de embedding"""
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dimensions.get(self.model, 1536)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_embedding(self, text: str) -> List[float]:
        """
        Gera embedding para um texto

        Args:
            text: Texto para gerar embedding

        Returns:
            Vetor de embedding
        """
        try:
            # Remove quebras de linha excessivas
            text = text.replace("\n", " ").strip()

            response = openai.embeddings.create(
                model=self.model,
                input=text
            )

            embedding = response.data[0].embedding
            return embedding

        except Exception as e:
            logger.error(f"Erro ao gerar embedding: {e}")
            raise

    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Gera embeddings em batch para melhor performance

        Args:
            texts: Lista de textos
            batch_size: Tamanho do batch
            show_progress: Mostrar barra de progresso

        Returns:
            Lista de vetores de embedding
        """
        all_embeddings = []

        # Processa em batches
        num_batches = (len(texts) + batch_size - 1) // batch_size

        iterator = range(num_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Gerando embeddings")

        for i in iterator:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]

            try:
                # Limpa textos
                cleaned_texts = [
                    text.replace("\n", " ").strip()
                    for text in batch_texts
                ]

                response = openai.embeddings.create(
                    model=self.model,
                    input=cleaned_texts
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                logger.debug(
                    f"Batch {i+1}/{num_batches}: "
                    f"{len(batch_embeddings)} embeddings gerados"
                )

            except Exception as e:
                logger.error(f"Erro no batch {i+1}: {e}")
                # Em caso de erro, tenta gerar individualmente
                for text in batch_texts:
                    try:
                        embedding = self.generate_embedding(text)
                        all_embeddings.append(embedding)
                    except Exception as inner_e:
                        logger.error(f"Erro ao gerar embedding individual: {inner_e}")
                        # Adiciona vetor zero em caso de erro
                        all_embeddings.append([0.0] * self.dimension)

        logger.info(f"Total de {len(all_embeddings)} embeddings gerados")
        return all_embeddings

    def create_vector_for_enriched_chunk(
        self,
        enriched_chunk: EnrichedChunk,
        use_enhanced_text: bool = True
    ) -> Dict[str, Any]:
        """
        Cria vetor completo para um chunk enriquecido

        Args:
            enriched_chunk: Chunk enriquecido
            use_enhanced_text: Se True, usa o texto enriquecido; senão usa o original

        Returns:
            Dicionário com formato do Pinecone: {id, values, metadata}
        """
        # Escolhe o texto a ser usado para embedding
        text_for_embedding = (
            enriched_chunk.enhanced_text if use_enhanced_text
            else enriched_chunk.original_chunk.text
        )

        # Gera embedding
        embedding = self.generate_embedding(text_for_embedding)

        # Prepara metadata rica
        metadata = {
            # Informações do chunk original
            "doc_id": enriched_chunk.original_chunk.doc_id,
            "chunk_index": enriched_chunk.original_chunk.chunk_index,
            "start_char": enriched_chunk.original_chunk.start_char,
            "end_char": enriched_chunk.original_chunk.end_char,
            "token_count": enriched_chunk.original_chunk.token_count,

            # Texto original (limitado para não exceder limites do Pinecone)
            "text": enriched_chunk.original_chunk.text[:1000],

            # Contexto gerado pelo LLM
            "contextual_summary": enriched_chunk.contextual_summary,
            "topic": enriched_chunk.topic,
            "key_concepts": ", ".join(enriched_chunk.key_concepts[:5]),
            "keywords": ", ".join(enriched_chunk.keywords[:10]),
            "questions": " | ".join(enriched_chunk.questions[:3]),

            # Metadata adicional do chunk original
            **{
                k: v for k, v in enriched_chunk.original_chunk.metadata.items()
                if isinstance(v, (str, int, float, bool))
            }
        }

        # Cria ID único
        vector_id = f"{enriched_chunk.original_chunk.doc_id}_{enriched_chunk.original_chunk.chunk_index}"

        return {
            "id": vector_id,
            "values": embedding,
            "metadata": metadata
        }

    def create_vectors_batch(
        self,
        enriched_chunks: List[EnrichedChunk],
        use_enhanced_text: bool = True,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Cria vetores para múltiplos chunks enriquecidos

        Args:
            enriched_chunks: Lista de chunks enriquecidos
            use_enhanced_text: Usar texto enriquecido
            show_progress: Mostrar progresso

        Returns:
            Lista de vetores no formato Pinecone
        """
        logger.info(f"Criando vetores para {len(enriched_chunks)} chunks enriquecidos")

        # Extrai textos para gerar embeddings em batch
        texts = [
            chunk.enhanced_text if use_enhanced_text else chunk.original_chunk.text
            for chunk in enriched_chunks
        ]

        # Gera embeddings em batch
        embeddings = self.generate_embeddings_batch(
            texts,
            show_progress=show_progress
        )

        # Cria vetores com metadata
        vectors = []
        for enriched_chunk, embedding in zip(enriched_chunks, embeddings):
            metadata = {
                "doc_id": enriched_chunk.original_chunk.doc_id,
                "chunk_index": enriched_chunk.original_chunk.chunk_index,
                "start_char": enriched_chunk.original_chunk.start_char,
                "end_char": enriched_chunk.original_chunk.end_char,
                "token_count": enriched_chunk.original_chunk.token_count,
                "text": enriched_chunk.original_chunk.text[:1000],
                "contextual_summary": enriched_chunk.contextual_summary,
                "topic": enriched_chunk.topic,
                "key_concepts": ", ".join(enriched_chunk.key_concepts[:5]),
                "keywords": ", ".join(enriched_chunk.keywords[:10]),
                "questions": " | ".join(enriched_chunk.questions[:3]),
                **{
                    k: v for k, v in enriched_chunk.original_chunk.metadata.items()
                    if isinstance(v, (str, int, float, bool))
                }
            }

            vector_id = f"{enriched_chunk.original_chunk.doc_id}_{enriched_chunk.original_chunk.chunk_index}"

            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })

        logger.info(f"Criados {len(vectors)} vetores com sucesso")
        return vectors

    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Gera embedding para uma query de busca

        Args:
            query: Texto da query

        Returns:
            Vetor de embedding
        """
        return self.generate_embedding(query)
