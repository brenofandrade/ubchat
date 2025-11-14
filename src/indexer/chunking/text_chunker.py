"""Estratégia de chunking inteligente para textos"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re
import tiktoken
from loguru import logger

from ..config import ChunkingSettings


class ChunkStrategy(Enum):
    """Estratégias de chunking disponíveis"""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    SENTENCE = "sentence"


@dataclass
class Chunk:
    """Representa um chunk de texto"""
    text: str
    chunk_index: int
    doc_id: str
    start_char: int
    end_char: int
    token_count: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Converte o chunk para dicionário"""
        return {
            "text": self.text,
            "chunk_index": self.chunk_index,
            "doc_id": self.doc_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "token_count": self.token_count,
            "metadata": self.metadata
        }


class TextChunker:
    """Classe para chunking inteligente de textos"""

    def __init__(
        self,
        settings: ChunkingSettings,
        strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
        encoding_name: str = "cl100k_base"
    ):
        """
        Inicializa o chunker

        Args:
            settings: Configurações de chunking
            strategy: Estratégia de chunking a usar
            encoding_name: Nome do encoding do tiktoken
        """
        self.settings = settings
        self.strategy = strategy
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Conta o número de tokens no texto"""
        return len(self.encoding.encode(text))

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Divide um documento em chunks

        Args:
            text: Texto do documento
            doc_id: ID do documento
            metadata: Metadata adicional

        Returns:
            Lista de chunks
        """
        if metadata is None:
            metadata = {}

        # Seleciona a estratégia de chunking
        if self.strategy == ChunkStrategy.FIXED_SIZE:
            return self._chunk_fixed_size(text, doc_id, metadata)
        elif self.strategy == ChunkStrategy.SEMANTIC:
            return self._chunk_semantic(text, doc_id, metadata)
        elif self.strategy == ChunkStrategy.RECURSIVE:
            return self._chunk_recursive(text, doc_id, metadata)
        elif self.strategy == ChunkStrategy.SENTENCE:
            return self._chunk_by_sentence(text, doc_id, metadata)
        else:
            raise ValueError(f"Estratégia desconhecida: {self.strategy}")

    def _chunk_fixed_size(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunking por tamanho fixo com overlap"""
        chunks = []
        chunk_size = self.settings.chunk_size
        overlap = self.settings.chunk_overlap

        start = 0
        chunk_index = 0

        while start < len(text):
            # Define o fim do chunk
            end = start + chunk_size

            # Ajusta para não cortar palavras
            if end < len(text):
                # Procura o último espaço antes do fim
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space

            chunk_text = text[start:end].strip()

            if chunk_text:
                token_count = self.count_tokens(chunk_text)

                chunk = Chunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    doc_id=doc_id,
                    start_char=start,
                    end_char=end,
                    token_count=token_count,
                    metadata={**metadata, "strategy": "fixed_size"}
                )

                chunks.append(chunk)
                chunk_index += 1

            # Move para o próximo chunk com overlap
            start = end - overlap if end - overlap > start else end

        logger.info(f"Documento {doc_id} dividido em {len(chunks)} chunks (fixed_size)")
        return chunks

    def _chunk_recursive(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Chunking recursivo que tenta manter estrutura semântica
        Tenta dividir por: parágrafos -> sentenças -> palavras
        """
        chunks = []
        separators = ["\n\n", "\n", ". ", " "]

        def _split_text(text: str, separators: List[str]) -> List[str]:
            """Divide o texto recursivamente usando os separadores"""
            if not separators or self.count_tokens(text) <= self.settings.chunk_size:
                return [text]

            separator = separators[0]
            remaining_separators = separators[1:]

            splits = text.split(separator)
            result = []
            current_chunk = []
            current_size = 0

            for split in splits:
                split_tokens = self.count_tokens(split)

                # Se o split é muito grande, divide recursivamente
                if split_tokens > self.settings.chunk_size:
                    if current_chunk:
                        result.append(separator.join(current_chunk))
                        current_chunk = []
                        current_size = 0

                    # Divide recursivamente
                    sub_splits = _split_text(split, remaining_separators)
                    result.extend(sub_splits)
                    continue

                # Verifica se adicionar o split excede o tamanho máximo
                if current_size + split_tokens > self.settings.chunk_size:
                    if current_chunk:
                        result.append(separator.join(current_chunk))
                        current_chunk = [split]
                        current_size = split_tokens
                else:
                    current_chunk.append(split)
                    current_size += split_tokens

            if current_chunk:
                result.append(separator.join(current_chunk))

            return result

        text_chunks = _split_text(text, separators)

        # Cria objetos Chunk
        char_position = 0
        for chunk_index, chunk_text in enumerate(text_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            token_count = self.count_tokens(chunk_text)
            start_char = text.find(chunk_text, char_position)
            end_char = start_char + len(chunk_text)

            chunk = Chunk(
                text=chunk_text,
                chunk_index=chunk_index,
                doc_id=doc_id,
                start_char=start_char,
                end_char=end_char,
                token_count=token_count,
                metadata={**metadata, "strategy": "recursive"}
            )

            chunks.append(chunk)
            char_position = end_char

        logger.info(f"Documento {doc_id} dividido em {len(chunks)} chunks (recursive)")
        return chunks

    def _chunk_by_sentence(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunking por sentenças"""
        # Regex simples para detectar fim de sentenças
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        char_position = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_tokens = self.count_tokens(sentence)

            # Se a sentença sozinha é muito grande, divide ela
            if sentence_tokens > self.settings.chunk_size:
                # Salva o chunk atual se houver
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    start_char = text.find(chunk_text, char_position)
                    end_char = start_char + len(chunk_text)

                    chunk = Chunk(
                        text=chunk_text,
                        chunk_index=chunk_index,
                        doc_id=doc_id,
                        start_char=start_char,
                        end_char=end_char,
                        token_count=current_tokens,
                        metadata={**metadata, "strategy": "sentence"}
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    char_position = end_char
                    current_chunk = []
                    current_tokens = 0

                # Divide a sentença grande
                words = sentence.split()
                temp_chunk = []
                temp_tokens = 0

                for word in words:
                    word_tokens = self.count_tokens(word)
                    if temp_tokens + word_tokens > self.settings.chunk_size:
                        if temp_chunk:
                            chunk_text = " ".join(temp_chunk)
                            start_char = text.find(chunk_text, char_position)
                            end_char = start_char + len(chunk_text)

                            chunk = Chunk(
                                text=chunk_text,
                                chunk_index=chunk_index,
                                doc_id=doc_id,
                                start_char=start_char,
                                end_char=end_char,
                                token_count=temp_tokens,
                                metadata={**metadata, "strategy": "sentence"}
                            )
                            chunks.append(chunk)
                            chunk_index += 1
                            char_position = end_char

                        temp_chunk = [word]
                        temp_tokens = word_tokens
                    else:
                        temp_chunk.append(word)
                        temp_tokens += word_tokens

                if temp_chunk:
                    current_chunk = temp_chunk
                    current_tokens = temp_tokens

                continue

            # Verifica se adicionar a sentença excede o limite
            if current_tokens + sentence_tokens > self.settings.chunk_size:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    start_char = text.find(chunk_text, char_position)
                    end_char = start_char + len(chunk_text)

                    chunk = Chunk(
                        text=chunk_text,
                        chunk_index=chunk_index,
                        doc_id=doc_id,
                        start_char=start_char,
                        end_char=end_char,
                        token_count=current_tokens,
                        metadata={**metadata, "strategy": "sentence"}
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    char_position = end_char

                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Adiciona o último chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            start_char = text.find(chunk_text, char_position)
            end_char = start_char + len(chunk_text)

            chunk = Chunk(
                text=chunk_text,
                chunk_index=chunk_index,
                doc_id=doc_id,
                start_char=start_char,
                end_char=end_char,
                token_count=current_tokens,
                metadata={**metadata, "strategy": "sentence"}
            )
            chunks.append(chunk)

        logger.info(f"Documento {doc_id} dividido em {len(chunks)} chunks (sentence)")
        return chunks

    def _chunk_semantic(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Chunking semântico - agrupa por tópicos similares
        (versão simplificada - pode ser melhorada com embeddings)
        """
        # Por enquanto, usa estratégia recursiva como fallback
        logger.warning("Chunking semântico usando estratégia recursiva como fallback")
        return self._chunk_recursive(text, doc_id, metadata)
