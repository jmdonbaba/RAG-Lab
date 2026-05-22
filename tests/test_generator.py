import pytest
from unittest.mock import MagicMock, patch

from src.generator import Generator, PROMPT_TEMPLATE


class TestGeneratorInit:
    def test_deepseek_defaults(self):
        gen = Generator(provider="deepseek")
        assert gen.provider == "deepseek"
        assert gen.model is not None

    def test_openai_defaults(self):
        gen = Generator(provider="openai")
        assert gen.provider == "openai"
        assert gen.model is not None

    def test_local_fallback(self):
        gen = Generator(provider="local")
        assert gen.provider == "local"
        assert gen.model is not None

    def test_custom_model(self):
        gen = Generator(provider="deepseek", model="custom-model")
        assert gen.model == "custom-model"


class TestFormatContext:
    def test_formats_multiple_docs(self):
        docs = [
            {"metadata": {"source": "lecture01.pdf"}, "content": "Content A"},
            {"metadata": {"source": "lecture02.pdf"}, "content": "Content B"},
        ]
        result = Generator._format_context(docs)
        assert "[Document 1 from lecture01.pdf]" in result
        assert "[Document 2 from lecture02.pdf]" in result
        assert "Content A" in result
        assert "Content B" in result

    def test_empty_docs_returns_empty_string(self):
        assert Generator._format_context([]) == ""


class TestFallbackResponse:
    def test_no_docs_returns_not_found(self):
        gen = Generator(provider="local")
        resp = gen.fallback_response("SVM", [])
        assert "未找到" in resp

    def test_with_docs_includes_sources(self):
        gen = Generator(provider="local")
        docs = [
            {"metadata": {"source": "svm.pdf"}, "content": "SVM content " + "x" * 300},
            {"metadata": {"source": "kernels.pdf"}, "content": "Kernel content " + "y" * 300},
        ]
        resp = gen.fallback_response("SVM", docs)
        assert "svm.pdf" in resp
        assert "kernels.pdf" in resp
        assert "SVM" in resp


class TestPromptTemplate:
    def test_template_renders_context_and_question(self):
        context = "Lecture notes on SVM"
        question = "What is a kernel?"
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        assert context in prompt
        assert question in prompt
        assert "机器学习助教" in prompt


class TestGenerate:
    def test_falls_back_to_config_key(self, monkeypatch):
        monkeypatch.setattr("src.generator.config.deepseek_api_key", "sk-test123")
        gen = Generator(provider="deepseek", api_key="")
        assert gen.api_key == "sk-test123"

    def test_returns_expected_structure(self, monkeypatch):
        monkeypatch.setattr("src.generator.Generator._call_api",
                            lambda self, prompt, temp: "Generated answer text.")
        gen = Generator(provider="deepseek", api_key="sk-test")
        docs = [{"metadata": {"source": "test.pdf"}, "content": "test content"}]
        result = gen.generate("test query", docs)

        assert result["query"] == "test query"
        assert result["answer"] == "Generated answer text."
        assert result["model"] == gen.model
        assert len(result["context_used"]) == 1
        assert result["sources"] == ["test.pdf"]

    def test_calls_api_with_context_in_prompt(self, monkeypatch):
        captured_prompt = []

        def fake_call_api(self, prompt, temp):
            captured_prompt.append(prompt)
            return "Answer."

        monkeypatch.setattr("src.generator.Generator._call_api", fake_call_api)
        gen = Generator(provider="deepseek", api_key="sk-test")
        docs = [{"metadata": {"source": "lecture01.pdf"}, "content": "SVM finds max-margin hyperplane."}]
        gen.generate("What is SVM?", docs)

        assert "SVM finds max-margin hyperplane" in captured_prompt[0]
        assert "What is SVM?" in captured_prompt[0]
        assert "[Document 1 from lecture01.pdf]" in captured_prompt[0]

    def test_raises_on_api_failure(self, monkeypatch):
        def fake_call_api(self, prompt, temp):
            raise RuntimeError("DeepSeek API call failed: connection refused")

        monkeypatch.setattr("src.generator.Generator._call_api", fake_call_api)
        gen = Generator(provider="deepseek", api_key="sk-test")
        docs = [{"metadata": {"source": "test.pdf"}, "content": "test content"}]

        with pytest.raises(RuntimeError, match="API call failed"):
            gen.generate("test query", docs)

    def test_local_provider_does_not_call_api(self, monkeypatch):
        """Local provider should route to _call_local, not _call_api."""
        monkeypatch.setattr("src.generator.Generator._call_local",
                            lambda self, prompt, temp: "Local model answer.")
        gen = Generator(provider="local")
        docs = [{"metadata": {"source": "test.pdf"}, "content": "test content"}]
        result = gen.generate("test query", docs)

        assert result["answer"] == "Local model answer."
        assert result["model"] is not None
