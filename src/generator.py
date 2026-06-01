import json
import logging
import time
from typing import List, Dict, Optional, Generator as Gen
from config import config

PROMPT_TEMPLATE = """你是一位知识渊博的机器学习助教。请根据提供的课程讲义内容回答用户的问题。仅使用上下文中的信息作答。如果上下文不足以回答问题，请如实说明。

## 上下文（课程讲义）
{context}

## 用户问题
{question}

## 回答要求
- 基于上述上下文内容作答
- 尽量准确，并在可能时注明信息来源的讲义章节
- 适当使用数学符号表达
- 回答简洁但全面

## 回答
"""


class Generator:
    """LLM-based answer generator with context grounding."""

    def __init__(self, provider: Optional[str] = None,
                 model: Optional[str] = None,
                 api_key: Optional[str] = None):
        self.provider = provider or config.llm_provider
        self.api_key = api_key

        if self.provider == "deepseek":
            self.model = model or config.deepseek_model
            if not self.api_key:
                self.api_key = config.deepseek_api_key
        elif self.provider == "openai":
            self.model = model or config.openai_model
            if not self.api_key:
                self.api_key = config.openai_api_key
        else:
            self.model = model or config.local_model

        self._local_model = None
        self._local_tokenizer = None

    def generate(self, query: str, context_docs: List[Dict],
                 temperature: float = 0.1) -> Dict:
        """Generate an answer grounded in retrieved context."""
        context_text = self._format_context(context_docs)
        prompt = PROMPT_TEMPLATE.format(context=context_text, question=query)

        if self.provider in ("deepseek", "openai"):
            answer = self._call_api(prompt, temperature)
        else:
            answer = self._call_local(prompt, temperature)

        return {
            "query": query,
            "answer": answer,
            "context_used": [d["content"][:200] for d in context_docs],
            "sources": [d["metadata"]["source"] for d in context_docs],
            "model": self.model,
        }

    def _call_api(self, prompt: str, temperature: float,
                  max_retries: int = 3) -> str:
        """Call LLM API with retry + exponential backoff."""
        from openai import OpenAI, RateLimitError, APIConnectionError, InternalServerError

        if self.provider == "deepseek":
            client = OpenAI(
                api_key=self.api_key,
                base_url=config.deepseek_base_url,
            )
        else:
            client = OpenAI(api_key=self.api_key)

        last_error = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1024,
                )
                return response.choices[0].message.content.strip()
            except (RateLimitError, APIConnectionError, InternalServerError) as e:
                last_error = e
                wait = 2 ** attempt
                logging.getLogger("rag_lab").warning(
                    "API call attempt %d/%d failed: %s. Retrying in %ds...",
                    attempt + 1, max_retries, e, wait)
                if attempt < max_retries - 1:
                    time.sleep(wait)
            except Exception as e:
                raise RuntimeError(
                    f"{self.provider.upper()} API call failed: {e}") from e

        raise RuntimeError(
            f"{self.provider.upper()} API call failed after {max_retries} "
            f"retries: {last_error}") from last_error

    def generate_stream(self, query: str, context_docs: List[Dict],
                        temperature: float = 0.1) -> Gen[str, None, None]:
        """Stream generated answer chunks via SSE."""
        context_text = self._format_context(context_docs)
        prompt = PROMPT_TEMPLATE.format(context=context_text, question=query)

        from openai import OpenAI

        if self.provider == "deepseek":
            client = OpenAI(
                api_key=self.api_key,
                base_url=config.deepseek_base_url,
            )
        else:
            client = OpenAI(api_key=self.api_key)

        stream = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def _call_local(self, prompt: str, temperature: float) -> str:
        """Run local model — loaded once and cached on the instance."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            if self._local_model is None:
                logging.getLogger("rag_lab").info(
                    "Loading local model: %s", config.local_model)
                self._local_tokenizer = AutoTokenizer.from_pretrained(
                    config.local_model, trust_remote_code=True)
                self._local_model = AutoModelForCausalLM.from_pretrained(
                    config.local_model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )

            inputs = self._local_tokenizer(
                prompt, return_tensors="pt").to(self._local_model.device)
            outputs = self._local_model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=temperature,
                do_sample=True,
            )
            return self._local_tokenizer.decode(
                outputs[0], skip_special_tokens=True)[len(prompt):].strip()
        except Exception as e:
            return f"[Local model unavailable: {e}]"

    def fallback_response(self, query: str, context_docs: List[Dict]) -> str:
        """Template-based fallback when no LLM is available."""
        if not context_docs:
            return "未找到相关的课程讲义内容来回答此问题。"

        lines = [
            f"基于 CMU 10-701 机器学习课程讲义，以下是与「{query}」最相关的内容：\n",
        ]
        for i, doc in enumerate(context_docs[:3], 1):
            src = doc["metadata"]["source"]
            lines.append(f"**来源 {i}** ({src}):")
            lines.append(doc["content"][:300])
            lines.append("")
        lines.append("---")
        lines.append("注意：此为仅检索模式的回复。接入 LLM 后可获得生成式回答。")
        return "\n".join(lines)

    @staticmethod
    def _format_context(docs: List[Dict]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            src = doc["metadata"]["source"]
            parts.append(f"[Document {i} from {src}]\n{doc['content']}")
        return "\n\n".join(parts)
