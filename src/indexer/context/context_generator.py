"""Gerador de contexto usando LLM para enriquecer chunks"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import openai
from anthropic import Anthropic
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from ..config import OpenAISettings, AnthropicSettings, ContextSettings
from ..chunking.text_chunker import Chunk


@dataclass
class EnrichedChunk:
    """Chunk enriquecido com contexto gerado por LLM"""
    original_chunk: Chunk
    contextual_summary: str
    key_concepts: List[str]
    keywords: List[str]
    topic: str
    questions: List[str] = field(default_factory=list)
    enhanced_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return {
            **self.original_chunk.to_dict(),
            "contextual_summary": self.contextual_summary,
            "key_concepts": self.key_concepts,
            "keywords": self.keywords,
            "topic": self.topic,
            "questions": self.questions,
            "enhanced_text": self.enhanced_text
        }


class ContextGenerator:
    """
    Gera contexto rico para chunks usando LLM

    Esta classe é o diferencial da estratégia de indexação, pois:
    1. Analisa cada chunk com LLM para extrair contexto semântico
    2. Identifica conceitos-chave e tópicos principais
    3. Gera perguntas que o chunk pode responder
    4. Cria um resumo contextual que melhora a recuperação
    """

    def __init__(
        self,
        openai_settings: Optional[OpenAISettings] = None,
        anthropic_settings: Optional[AnthropicSettings] = None,
        context_settings: Optional[ContextSettings] = None,
        use_provider: str = "openai"
    ):
        """
        Inicializa o gerador de contexto

        Args:
            openai_settings: Configurações OpenAI
            anthropic_settings: Configurações Anthropic
            context_settings: Configurações de contexto
            use_provider: Provider a usar ("openai" ou "anthropic")
        """
        self.openai_settings = openai_settings
        self.anthropic_settings = anthropic_settings
        self.context_settings = context_settings or ContextSettings()
        self.use_provider = use_provider

        # Inicializa os clientes
        if use_provider == "openai" and openai_settings:
            openai.api_key = openai_settings.api_key
            self.openai_model = openai_settings.model
        elif use_provider == "anthropic" and anthropic_settings:
            self.anthropic_client = Anthropic(api_key=anthropic_settings.api_key)
            self.anthropic_model = anthropic_settings.model

        # Templates de prompts
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        """Carrega os templates de prompts"""
        return {
            "default": """Analise o seguinte trecho de um documento e forneça:

1. Um resumo contextual em 2-3 frases que capture a essência e o propósito do texto
2. Uma lista de 3-5 conceitos-chave principais
3. Uma lista de 5-8 palavras-chave relevantes
4. O tópico principal em uma única frase
5. 2-3 perguntas que este trecho pode responder

TEXTO:
{text}

Responda APENAS no seguinte formato JSON:
{{
  "contextual_summary": "resumo aqui",
  "key_concepts": ["conceito1", "conceito2", "conceito3"],
  "keywords": ["palavra1", "palavra2", "palavra3"],
  "topic": "tópico principal",
  "questions": ["pergunta1?", "pergunta2?"]
}}""",

            "detailed": """Como um especialista em análise de documentos, analise profundamente o seguinte texto:

TEXTO:
{text}

CONTEXTO DO DOCUMENTO: {doc_context}

Forneça uma análise detalhada incluindo:
1. RESUMO CONTEXTUAL: Explique o propósito e significado deste trecho no contexto maior
2. CONCEITOS-CHAVE: Identifique os 3-5 conceitos mais importantes e relevantes
3. PALAVRAS-CHAVE: Liste 5-8 termos que melhor representam o conteúdo
4. TÓPICO PRINCIPAL: Categorize o assunto principal em uma frase clara
5. PERGUNTAS RESPONDIDAS: Que perguntas específicas este texto pode responder?

Responda em formato JSON:
{{
  "contextual_summary": "...",
  "key_concepts": [...],
  "keywords": [...],
  "topic": "...",
  "questions": [...]
}}""",

            "technical": """Analise este trecho técnico com foco em:
- Terminologia técnica específica
- Conceitos e princípios importantes
- Relações entre componentes/conceitos
- Aplicações práticas

TEXTO:
{text}

Formato de resposta JSON:
{{
  "contextual_summary": "...",
  "key_concepts": [...],
  "keywords": [...],
  "topic": "...",
  "questions": [...]
}}"""
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _call_openai(self, prompt: str) -> Dict[str, Any]:
        """Chama a API da OpenAI com retry"""
        try:
            response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um assistente especializado em análise e contextualização de documentos. Sempre responda em formato JSON válido."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            return eval(content)  # Parse JSON

        except Exception as e:
            logger.error(f"Erro ao chamar OpenAI: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _call_anthropic(self, prompt: str) -> Dict[str, Any]:
        """Chama a API da Anthropic com retry"""
        try:
            response = self.anthropic_client.messages.create(
                model=self.anthropic_model,
                max_tokens=1024,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            content = response.content[0].text
            # Extrai JSON da resposta
            import json
            import re

            # Procura por JSON na resposta
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("Não foi possível extrair JSON da resposta")

        except Exception as e:
            logger.error(f"Erro ao chamar Anthropic: {e}")
            raise

    def generate_context_for_chunk(
        self,
        chunk: Chunk,
        doc_context: Optional[str] = None,
        template: str = "default"
    ) -> EnrichedChunk:
        """
        Gera contexto enriquecido para um único chunk

        Args:
            chunk: Chunk a ser analisado
            doc_context: Contexto adicional do documento
            template: Template de prompt a usar

        Returns:
            Chunk enriquecido com contexto
        """
        # Prepara o prompt
        prompt_template = self.prompts.get(template, self.prompts["default"])
        prompt = prompt_template.format(
            text=chunk.text,
            doc_context=doc_context or "Não especificado"
        )

        # Chama o LLM
        try:
            if self.use_provider == "openai":
                result = self._call_openai(prompt)
            else:
                result = self._call_anthropic(prompt)

            # Cria texto enriquecido combinando contexto com o texto original
            enhanced_text = self._create_enhanced_text(chunk, result)

            enriched_chunk = EnrichedChunk(
                original_chunk=chunk,
                contextual_summary=result.get("contextual_summary", ""),
                key_concepts=result.get("key_concepts", []),
                keywords=result.get("keywords", []),
                topic=result.get("topic", ""),
                questions=result.get("questions", []),
                enhanced_text=enhanced_text
            )

            logger.debug(
                f"Contexto gerado para chunk {chunk.chunk_index} "
                f"do documento {chunk.doc_id}"
            )

            return enriched_chunk

        except Exception as e:
            logger.error(
                f"Erro ao gerar contexto para chunk {chunk.chunk_index}: {e}"
            )
            # Retorna chunk com contexto básico em caso de erro
            return EnrichedChunk(
                original_chunk=chunk,
                contextual_summary="Erro ao gerar contexto",
                key_concepts=[],
                keywords=[],
                topic="Desconhecido",
                questions=[],
                enhanced_text=chunk.text
            )

    def _create_enhanced_text(
        self,
        chunk: Chunk,
        context: Dict[str, Any]
    ) -> str:
        """
        Cria texto enriquecido combinando o chunk original com o contexto

        Esta é uma estratégia chave: o texto enriquecido será usado para gerar
        embeddings mais informativos e contextualizados
        """
        # Formato do texto enriquecido
        enhanced = f"""CONTEXTO: {context.get('contextual_summary', '')}

TÓPICO: {context.get('topic', '')}

CONCEITOS-CHAVE: {', '.join(context.get('key_concepts', []))}

CONTEÚDO:
{chunk.text}

PERGUNTAS RELACIONADAS:
{chr(10).join(f"- {q}" for q in context.get('questions', []))}

PALAVRAS-CHAVE: {', '.join(context.get('keywords', []))}"""

        return enhanced

    def generate_contexts_batch(
        self,
        chunks: List[Chunk],
        doc_context: Optional[str] = None,
        template: str = "default",
        show_progress: bool = True
    ) -> List[EnrichedChunk]:
        """
        Gera contextos para múltiplos chunks em batch

        Args:
            chunks: Lista de chunks
            doc_context: Contexto do documento
            template: Template de prompt
            show_progress: Mostrar barra de progresso

        Returns:
            Lista de chunks enriquecidos
        """
        enriched_chunks = []

        iterator = tqdm(chunks, desc="Gerando contextos") if show_progress else chunks

        for chunk in iterator:
            enriched_chunk = self.generate_context_for_chunk(
                chunk,
                doc_context,
                template
            )
            enriched_chunks.append(enriched_chunk)

        logger.info(f"Contextos gerados para {len(enriched_chunks)} chunks")
        return enriched_chunks

    def generate_document_summary(
        self,
        full_text: str,
        max_length: int = 500
    ) -> str:
        """
        Gera um resumo do documento completo para usar como contexto

        Args:
            full_text: Texto completo do documento
            max_length: Tamanho máximo do resumo

        Returns:
            Resumo do documento
        """
        # Trunca o texto se for muito longo
        if len(full_text) > 10000:
            text_sample = full_text[:5000] + "\n...\n" + full_text[-5000:]
        else:
            text_sample = full_text

        prompt = f"""Gere um resumo conciso (máximo {max_length} caracteres) deste documento,
capturando seu propósito principal, tópicos abordados e contexto geral:

{text_sample}

Responda apenas com o resumo, sem formatação adicional."""

        try:
            if self.use_provider == "openai":
                response = openai.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Você é um especialista em sumarização de documentos."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                summary = response.choices[0].message.content.strip()
            else:
                response = self.anthropic_client.messages.create(
                    model=self.anthropic_model,
                    max_tokens=200,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                summary = response.content[0].text.strip()

            logger.info(f"Resumo do documento gerado: {len(summary)} caracteres")
            return summary

        except Exception as e:
            logger.error(f"Erro ao gerar resumo do documento: {e}")
            return "Documento sem resumo disponível"
