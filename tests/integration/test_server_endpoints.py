# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for oMLX server endpoints.

Tests the FastAPI endpoints using TestClient with mocked EnginePool and Engine
to verify request/response formats without loading actual models.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from fastapi.testclient import TestClient

from omlx.api.responses_utils import ResponseStore
from omlx.engine.embedding import EmbeddingEngine
from omlx.engine.reranker import RerankerEngine

TEST_API_KEY = "test-api-key"


@dataclass
class MockEmbeddingOutput:
    """Mock embedding output for testing."""

    embeddings: List[List[float]] = field(
        default_factory=lambda: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )
    total_tokens: int = 10
    dimensions: int = 3


@dataclass
class MockRerankOutput:
    """Mock rerank output for testing."""

    scores: List[float] = field(default_factory=lambda: [0.9, 0.5, 0.3])
    indices: List[int] = field(default_factory=lambda: [0, 1, 2])
    total_tokens: int = 50


@dataclass
class MockGenerationOutput:
    """Mock generation output for testing."""

    text: str = "Hello, I am a helpful assistant."
    tokens: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    prompt_tokens: int = 10
    completion_tokens: int = 5
    finish_reason: str = "stop"
    new_text: str = ""
    finished: bool = True
    tool_calls: Optional[List[Dict[str, Any]]] = None
    cached_tokens: int = 0


class MockEmbeddingEngineImpl(EmbeddingEngine):
    """Mock embedding engine for testing that inherits from EmbeddingEngine."""

    def __init__(self, model_name: str = "test-embedding-model"):
        # Don't call super().__init__ to avoid loading real model
        self._model_name = model_name
        self._model = None  # Set as None but present

    @property
    def model_name(self) -> str:
        return self._model_name

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def embed(self, texts, **kwargs) -> MockEmbeddingOutput:
        return MockEmbeddingOutput(
            embeddings=[[0.1, 0.2, 0.3] for _ in texts],
            total_tokens=len(texts) * 5,
            dimensions=3,
        )

    def get_stats(self) -> Dict[str, Any]:
        return {"model_name": self._model_name, "loaded": True}


class MockRerankerEngineImpl(RerankerEngine):
    """Mock reranker engine for testing that inherits from RerankerEngine."""

    def __init__(self, model_name: str = "test-reranker-model"):
        # Don't call super().__init__ to avoid loading real model
        self._model_name = model_name
        self._model = None  # Set as None but present

    @property
    def model_name(self) -> str:
        return self._model_name

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def rerank(
        self, query: str, documents: List[str], top_n: Optional[int] = None, **kwargs
    ) -> MockRerankOutput:
        n_docs = len(documents)
        scores = [0.9 - i * 0.2 for i in range(n_docs)]
        indices = list(range(n_docs))
        if top_n:
            indices = indices[:top_n]
        return MockRerankOutput(
            scores=scores,
            indices=indices,
            total_tokens=n_docs * 20,
        )

    def get_stats(self) -> Dict[str, Any]:
        return {"model_name": self._model_name, "loaded": True}


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.eos_token_id = 2

    def encode(self, text: str) -> List[int]:
        # Simple simulation: split by words
        return [100 + i for i, _ in enumerate(text.split())]

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        return f"<decoded:{len(tokens)} tokens>"

    def apply_chat_template(
        self, messages: List[Dict], tokenize: bool = False, **kwargs
    ) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)


class MockBaseEngine:
    """Mock LLM engine for testing."""

    def __init__(self, model_name: str = "test-llm-model"):
        self._model_name = model_name
        self._tokenizer = MockTokenizer()
        self._model_type = "llama"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model_type(self) -> Optional[str]:
        return self._model_type

    async def generate(self, prompt: str, **kwargs) -> MockGenerationOutput:
        return MockGenerationOutput(text="Generated response.")

    async def stream_generate(self, prompt: str, **kwargs):
        yield MockGenerationOutput(
            text="Hello",
            new_text="Hello",
            finished=False,
        )
        yield MockGenerationOutput(
            text="Hello world",
            new_text=" world",
            finished=True,
            finish_reason="stop",
        )

    def count_chat_tokens(self, messages: List[Dict], tools=None, chat_template_kwargs=None) -> int:
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False)
        return len(self._tokenizer.encode(prompt))

    async def chat(self, messages: List[Dict], **kwargs) -> MockGenerationOutput:
        return MockGenerationOutput(text="Chat response.")

    async def stream_chat(self, messages: List[Dict], **kwargs):
        yield MockGenerationOutput(
            text="Hello",
            new_text="Hello",
            finished=False,
        )
        yield MockGenerationOutput(
            text="Hello from chat",
            new_text=" from chat",
            finished=True,
            finish_reason="stop",
        )


class RecordingResponsesEngine(MockBaseEngine):
    """Mock engine that records request messages across /v1/responses calls."""

    def __init__(self, outputs: Optional[List[MockGenerationOutput]] = None):
        super().__init__()
        self._outputs = list(outputs or [])
        self.recorded_messages: List[List[Dict[str, Any]]] = []
        self._model_type = "gpt_oss"

    async def chat(self, messages: List[Dict], **kwargs) -> MockGenerationOutput:
        self.recorded_messages.append(messages)
        if self._outputs:
            return self._outputs.pop(0)
        return MockGenerationOutput(text="Chat response.")


class MockEnginePool:
    """Mock engine pool for testing."""

    def __init__(
        self,
        llm_engine: Optional[MockBaseEngine] = None,
        embedding_engine: Optional[MockEmbeddingEngineImpl] = None,
        reranker_engine: Optional[MockRerankerEngineImpl] = None,
    ):
        self._llm_engine = llm_engine or MockBaseEngine()
        self._embedding_engine = embedding_engine
        self._reranker_engine = reranker_engine
        self._models = [
            {"id": "test-model", "loaded": True, "pinned": False, "size": 1000000}
        ]

    @property
    def model_count(self) -> int:
        return len(self._models)

    @property
    def loaded_model_count(self) -> int:
        return sum(1 for m in self._models if m["loaded"])

    @property
    def max_model_memory(self) -> int:
        return 32 * 1024 * 1024 * 1024  # 32GB

    @property
    def current_model_memory(self) -> int:
        return 1000000

    def get_entry(self, model_id: str):
        return None

    def resolve_model_id(self, model_id_or_alias, settings_manager=None):
        return model_id_or_alias

    def acquire_engine(self, model_id: str) -> None:
        pass

    def release_engine(self, model_id: str) -> None:
        pass

    def ensure_engine_alive(self, model_id: str, engine_ref) -> None:
        # Tests never race with the process memory enforcer, so treat
        # every engine reference as live.
        pass

    def get_model_ids(self) -> List[str]:
        return [m["id"] for m in self._models]

    def get_loaded_model_ids(self) -> List[str]:
        return [m["id"] for m in self._models if m.get("loaded")]

    def get_status(self) -> Dict[str, Any]:
        return {
            "models": self._models,
            "loaded_count": self.loaded_model_count,
            "max_model_memory": self.max_model_memory,
        }

    async def get_engine(self, model_id: str):
        # Return appropriate engine based on model name pattern
        if "embed" in model_id.lower():
            if self._embedding_engine:
                return self._embedding_engine
            raise ValueError(f"No embedding engine for {model_id}")
        elif "rerank" in model_id.lower():
            if self._reranker_engine:
                return self._reranker_engine
            raise ValueError(f"No reranker engine for {model_id}")
        return self._llm_engine


@pytest.fixture
def mock_llm_engine():
    """Create a mock LLM engine."""
    return MockBaseEngine()


@pytest.fixture
def mock_embedding_engine():
    """Create a mock embedding engine."""
    return MockEmbeddingEngineImpl()


@pytest.fixture
def mock_reranker_engine():
    """Create a mock reranker engine."""
    return MockRerankerEngineImpl()


@pytest.fixture
def mock_engine_pool(mock_llm_engine, mock_embedding_engine, mock_reranker_engine):
    """Create a mock engine pool."""
    return MockEnginePool(
        llm_engine=mock_llm_engine,
        embedding_engine=mock_embedding_engine,
        reranker_engine=mock_reranker_engine,
    )


@pytest.fixture
def client(mock_engine_pool):
    """Create a test client with mocked server state."""
    from omlx.server import app, _server_state

    # Store original state
    original_pool = _server_state.engine_pool
    original_default = _server_state.default_model
    original_api_key = _server_state.api_key

    # Set mock state
    _server_state.engine_pool = mock_engine_pool
    _server_state.default_model = "test-model"
    _server_state.api_key = TEST_API_KEY

    yield TestClient(app, headers={"Authorization": f"Bearer {TEST_API_KEY}"})

    # Restore original state
    _server_state.engine_pool = original_pool
    _server_state.default_model = original_default
    _server_state.api_key = original_api_key


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_healthy_status(self, client):
        """Test that health endpoint returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_contains_required_fields(self, client):
        """Test that health response contains required fields."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "default_model" in data
        assert "engine_pool" in data

    def test_health_engine_pool_info(self, client):
        """Test that health response contains engine pool info."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        pool_info = data["engine_pool"]
        assert "model_count" in pool_info
        assert "loaded_count" in pool_info
        assert "max_model_memory" in pool_info
        assert "current_model_memory" in pool_info


class TestModelsEndpoint:
    """Tests for the /v1/models endpoint."""

    def test_models_returns_list(self, client):
        """Test that models endpoint returns a list."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data

    def test_models_format(self, client):
        """Test that model entries have correct format."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        if data["data"]:
            model = data["data"][0]
            assert "id" in model
            assert "object" in model


class TestResponsesEndpoint:
    def test_response_endpoint_recovers_tool_call_from_thinking(self, tmp_path):
        from omlx.server import app, _server_state

        state_dir = tmp_path / "response-state"
        engine = RecordingResponsesEngine(outputs=[
            MockGenerationOutput(
                text=(
                    "<think>Need to inspect first."
                    '<tool_call>{"name":"exec_command","arguments":{"cmd":"ls"}}</tool_call>'
                    "Then continue.</think>"
                ),
                finish_reason="stop",
            ),
        ])
        pool = MockEnginePool(llm_engine=engine)

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_store = _server_state.responses_store
        original_api_key = _server_state.api_key
        try:
            _server_state.engine_pool = pool
            _server_state.default_model = "test-model"
            _server_state.responses_store = ResponseStore(state_dir=state_dir)
            _server_state.api_key = TEST_API_KEY
            client = TestClient(app, headers={"Authorization": f"Bearer {TEST_API_KEY}"})

            response = client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "Explore the code",
                    "tools": [{
                        "type": "function",
                        "name": "exec_command",
                        "description": "Run a shell command",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    }],
                },
            )
            assert response.status_code == 200

            output_items = response.json()["output"]
            message_items = [item for item in output_items if item["type"] == "message"]
            function_items = [item for item in output_items if item["type"] == "function_call"]

            assert len(message_items) == 1
            assert message_items[0]["content"][0]["text"] == ""
            assert "<tool_call>" not in message_items[0]["content"][0]["text"]
            assert len(function_items) == 1
            assert function_items[0]["name"] == "exec_command"
            assert function_items[0]["arguments"] == '{"cmd": "ls"}'
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.responses_store = original_store
            _server_state.api_key = original_api_key

    def test_previous_response_id_persists_across_store_restart(self, tmp_path):
        from omlx.server import app, _server_state

        state_dir = tmp_path / "response-state"
        engine = RecordingResponsesEngine(outputs=[
            MockGenerationOutput(
                text="",
                finish_reason="tool_calls",
                tool_calls=[{
                    "id": "call_123",
                    "name": "exec_command",
                    "arguments": '{"cmd":"ls"}',
                }],
            ),
            MockGenerationOutput(text="Done.", finish_reason="stop"),
        ])
        pool = MockEnginePool(llm_engine=engine)

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_store = _server_state.responses_store
        original_api_key = _server_state.api_key
        try:
            _server_state.engine_pool = pool
            _server_state.default_model = "test-model"
            _server_state.responses_store = ResponseStore(state_dir=state_dir)
            _server_state.api_key = TEST_API_KEY
            client = TestClient(app, headers={"Authorization": f"Bearer {TEST_API_KEY}"})

            first = client.post(
                "/v1/responses",
                json={"model": "test-model", "input": "Explore the code"},
            )
            assert first.status_code == 200
            first_id = first.json()["id"]

            # Simulate a restart by rebuilding the store from disk.
            _server_state.responses_store = ResponseStore(state_dir=state_dir)

            second = client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "previous_response_id": first_id,
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "call_123",
                            "output": "file1.txt\nfile2.txt",
                        }
                    ],
                },
            )
            assert second.status_code == 200

            replayed = engine.recorded_messages[1]
            assert replayed[0] == {"role": "user", "content": "Explore the code"}
            assert replayed[1]["role"] == "assistant"
            assert replayed[1]["tool_calls"][0]["id"] == "call_123"
            assert replayed[2] == {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "file1.txt\nfile2.txt",
            }
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.responses_store = original_store
            _server_state.api_key = original_api_key

    def test_missing_previous_response_id_returns_404(self, tmp_path):
        from omlx.server import app, _server_state

        engine = RecordingResponsesEngine(outputs=[MockGenerationOutput(text="Done.")])
        pool = MockEnginePool(llm_engine=engine)

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model
        original_store = _server_state.responses_store
        original_api_key = _server_state.api_key
        try:
            _server_state.engine_pool = pool
            _server_state.default_model = "test-model"
            _server_state.responses_store = ResponseStore(
                state_dir=tmp_path / "response-state"
            )
            _server_state.api_key = TEST_API_KEY
            client = TestClient(app, headers={"Authorization": f"Bearer {TEST_API_KEY}"})

            response = client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "previous_response_id": "resp_missing",
                    "input": "Continue",
                },
            )
            assert response.status_code == 404
        finally:
            _server_state.engine_pool = original_pool
            _server_state.default_model = original_default
            _server_state.responses_store = original_store
            _server_state.api_key = original_api_key


class TestModelsStatusEndpoint:
    """Tests for the /v1/models/status endpoint."""

    def test_models_status_returns_details(self, client):
        """Test that models status returns detailed info."""
        response = client.get("/v1/models/status")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data


class TestCompletionEndpoint:
    """Tests for the /v1/completions endpoint."""

    def test_completion_basic_request(self, client):
        """Test basic completion request."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Hello, world!",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "text" in data["choices"][0]

    def test_completion_response_format(self, client):
        """Test completion response has correct format."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Test prompt",
                "max_tokens": 100,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "text_completion"
        assert "model" in data
        assert "choices" in data
        assert "usage" in data

    def test_completion_with_list_prompt(self, client):
        """Test completion with list of prompts."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": ["First prompt", "Second prompt"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data


class TestChatCompletionEndpoint:
    """Tests for the /v1/chat/completions endpoint."""

    def test_chat_completion_basic(self, client):
        """Test basic chat completion request."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0

    def test_chat_completion_response_format(self, client):
        """Test chat completion response format."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi!"},
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert "model" in data
        assert "choices" in data
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "usage" in data

    def test_chat_completion_with_parameters(self, client):
        """Test chat completion with sampling parameters."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Test"}],
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 256,
            },
        )

        assert response.status_code == 200

    def test_chat_completion_sanitizes_reasoning_tool_call_markup(self, client, mock_llm_engine):
        """Thinking-only tool calls should become structured tool_calls without leaked markup."""
        mock_llm_engine.chat = AsyncMock(return_value=MockGenerationOutput(
            text=(
                "<think>Need to inspect first."
                '<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>'
                "Then continue.</think>"
            ),
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="stop",
            finished=True,
        ))

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }],
            },
        )

        assert response.status_code == 200
        data = response.json()
        message = data["choices"][0]["message"]

        assert message["reasoning_content"] == "Need to inspect first.Then continue."
        assert "<tool_call>" not in message["reasoning_content"]
        assert len(message["tool_calls"]) == 1
        assert message["tool_calls"][0]["function"]["name"] == "get_weather"
        assert message["tool_calls"][0]["function"]["arguments"] == '{"city": "SF"}'
        assert data["choices"][0]["finish_reason"] == "tool_calls"


class TestAnthropicMessagesEndpoint:
    """Tests for the /v1/messages endpoint (Anthropic format)."""

    def test_anthropic_messages_basic(self, client):
        """Test basic Anthropic messages request."""
        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"

    def test_anthropic_messages_response_format(self, client):
        """Test Anthropic messages response format."""
        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hi there!"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "content" in data
        assert "usage" in data
        assert "input_tokens" in data["usage"]
        assert "output_tokens" in data["usage"]

    def test_anthropic_messages_with_system(self, client):
        """Test Anthropic messages with system prompt."""
        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 1024,
                "system": "You are a helpful assistant.",
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )

        assert response.status_code == 200

    def test_anthropic_messages_sanitize_thinking_tool_call_markup(self, client, mock_llm_engine):
        """Anthropic thinking blocks should not expose raw tool-call markup."""
        mock_llm_engine.chat = AsyncMock(return_value=MockGenerationOutput(
            text=(
                "<think>Need to inspect first."
                '<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>'
                "Then continue.</think>"
            ),
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="stop",
            finished=True,
        ))

        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": [{
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }],
            },
        )

        assert response.status_code == 200
        data = response.json()
        thinking_blocks = [block for block in data["content"] if block["type"] == "thinking"]
        tool_use_blocks = [block for block in data["content"] if block["type"] == "tool_use"]

        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "Need to inspect first.Then continue."
        assert "<tool_call>" not in thinking_blocks[0]["thinking"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["name"] == "get_weather"
        assert tool_use_blocks[0]["input"] == {"city": "SF"}
        assert data["stop_reason"] == "tool_use"


class TestEmbeddingsEndpoint:
    """Tests for the /v1/embeddings endpoint."""

    def test_embeddings_single_input(self, client, mock_engine_pool):
        """Test embeddings with single input."""
        mock_engine_pool._models.append(
            {"id": "test-embed-model", "loaded": True, "pinned": False, "size": 500000}
        )

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "test-embed-model",
                "input": "Hello, world!",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) == 1
        assert data["data"][0]["object"] == "embedding"

    def test_embeddings_multiple_inputs(self, client, mock_engine_pool):
        """Test embeddings with multiple inputs."""
        mock_engine_pool._models.append(
            {"id": "test-embed-model", "loaded": True, "pinned": False, "size": 500000}
        )

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "test-embed-model",
                "input": ["First text", "Second text"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2

    def test_embeddings_response_format(self, client, mock_engine_pool):
        """Test embeddings response format."""
        mock_engine_pool._models.append(
            {"id": "test-embed-model", "loaded": True, "pinned": False, "size": 500000}
        )

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "test-embed-model",
                "input": "Test text",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "model" in data
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]
        assert "embedding" in data["data"][0]
        assert isinstance(data["data"][0]["embedding"], list)

    def test_embeddings_structured_items_input(self, client, mock_engine_pool):
        """Test embeddings with structured multimodal items."""
        mock_engine_pool._models.append(
            {"id": "test-embed-model", "loaded": True, "pinned": False, "size": 500000}
        )

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "test-embed-model",
                "items": [
                    {"text": "hello"},
                    {"image": "https://example.com/image.jpg"},
                    {
                        "text": "hello",
                        "image": "https://example.com/image.jpg",
                    },
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 3

    def test_embeddings_rejects_mixed_input_sources(self, client, mock_engine_pool):
        """Test embeddings rejects input and items together."""
        mock_engine_pool._models.append(
            {"id": "test-embed-model", "loaded": True, "pinned": False, "size": 500000}
        )

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "test-embed-model",
                "input": "hello",
                "items": [{"text": "hello"}],
            },
        )

        assert response.status_code == 422


class TestRerankEndpoint:
    """Tests for the /v1/rerank endpoint."""

    def test_rerank_basic(self, client, mock_engine_pool):
        """Test basic rerank request."""
        mock_engine_pool._models.append(
            {
                "id": "test-rerank-model",
                "loaded": True,
                "pinned": False,
                "size": 500000,
            }
        )

        response = client.post(
            "/v1/rerank",
            json={
                "model": "test-rerank-model",
                "query": "What is machine learning?",
                "documents": [
                    "ML is a subset of AI.",
                    "The weather is nice today.",
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2

    def test_rerank_with_top_n(self, client, mock_engine_pool):
        """Test rerank with top_n parameter."""
        mock_engine_pool._models.append(
            {
                "id": "test-rerank-model",
                "loaded": True,
                "pinned": False,
                "size": 500000,
            }
        )

        response = client.post(
            "/v1/rerank",
            json={
                "model": "test-rerank-model",
                "query": "Test query",
                "documents": ["Doc 1", "Doc 2", "Doc 3"],
                "top_n": 2,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2

    def test_rerank_response_format(self, client, mock_engine_pool):
        """Test rerank response format."""
        mock_engine_pool._models.append(
            {
                "id": "test-rerank-model",
                "loaded": True,
                "pinned": False,
                "size": 500000,
            }
        )

        response = client.post(
            "/v1/rerank",
            json={
                "model": "test-rerank-model",
                "query": "Test",
                "documents": ["Document 1"],
                "return_documents": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "model" in data
        assert "results" in data
        result = data["results"][0]
        assert "index" in result
        assert "relevance_score" in result
        assert "document" in result


class TestTokenCountEndpoint:
    """Tests for the /v1/messages/count_tokens endpoint."""

    def test_token_count_basic(self, client):
        """Test basic token counting."""
        response = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello world"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "input_tokens" in data
        assert isinstance(data["input_tokens"], int)

    def test_token_count_with_system(self, client):
        """Test token counting with system prompt."""
        response = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "test-model",
                "system": "You are helpful.",
                "messages": [{"role": "user", "content": "Hi!"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "input_tokens" in data


class TestMCPEndpoints:
    """Tests for MCP-related endpoints."""

    def test_mcp_tools_empty(self, client):
        """Test MCP tools endpoint when no MCP configured."""
        response = client.get("/v1/mcp/tools")

        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert "count" in data
        assert data["count"] == 0

    def test_mcp_servers_empty(self, client):
        """Test MCP servers endpoint when no MCP configured."""
        response = client.get("/v1/mcp/servers")

        assert response.status_code == 200
        data = response.json()
        assert "servers" in data

    def test_mcp_execute_no_config(self, client):
        """Test MCP execute fails when not configured."""
        response = client.post(
            "/v1/mcp/execute",
            json={
                "tool_name": "test_tool",
                "arguments": {},
            },
        )

        # Should return 503 when MCP not configured
        assert response.status_code == 503


class TestErrorHandling:
    """Tests for error handling in endpoints."""

    def test_missing_model(self, client):
        """Test error when model is not specified."""
        # For Anthropic endpoint, missing model should raise validation error
        response = client.post(
            "/v1/messages",
            json={
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 422  # Validation error

    def test_empty_messages(self, client):
        """Test error when messages is empty."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [],
            },
        )

        # Empty messages may be allowed or raise error depending on implementation
        # Just verify we get a response
        assert response.status_code in [200, 400, 422]

    def test_invalid_request_format(self, client):
        """Test error for invalid request format."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "invalid_field": "test",
            },
        )

        assert response.status_code == 422


class TestRequestAbortedErrorHandling:
    """Tests for the FastAPI exception handler that translates
    RequestAbortedError (raised by EngineCore.generate when the
    process memory enforcer called abort_all_requests() mid-call)
    into HTTP 503 with an OpenAI-compatible error body.
    """

    def test_chat_completions_abort_returns_503(self, client, mock_llm_engine):
        """Non-streaming chat completions: engine.chat raising
        RequestAbortedError must surface as HTTP 503 with an
        OpenAI-shaped error envelope."""
        from omlx.exceptions import RequestAbortedError

        mock_llm_engine.chat = AsyncMock(
            side_effect=RequestAbortedError(
                "Request aborted: process memory limit exceeded. "
                "Increase --max-process-memory or reduce context size."
            )
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 503
        data = response.json()
        # OpenAI-style error envelope from _openai_error_body
        assert "error" in data
        assert "process memory limit exceeded" in data["error"]["message"]

    def test_completions_abort_returns_503(self, client, mock_llm_engine):
        """Non-streaming /v1/completions: engine.generate raising
        RequestAbortedError must also surface as HTTP 503."""
        from omlx.exceptions import RequestAbortedError

        mock_llm_engine.generate = AsyncMock(
            side_effect=RequestAbortedError(
                "Request aborted: process memory limit exceeded."
            )
        )

        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Hello",
            },
        )

        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "process memory limit exceeded" in data["error"]["message"]


class TestEngineEvictedErrorHandling:
    """Tests for the handler-side hardening that translates
    EngineEvictedError into HTTP 503.

    Two entry points into server.get_engine can raise this exception:

    1. `pool.get_engine(...)` itself raises — e.g. if the pool's
       internal state detected a stale entry during load resolution.
       Caught inside the existing try/except in server.get_engine at
       the `except EngineEvictedError` branch.

    2. `pool.ensure_engine_alive(model_id, engine)` raises after
       `pool.get_engine(...)` returned successfully. This is the
       specific race the hardening was added for: the process memory
       enforcer nulled `entry.engine` between get_engine's last yield
       and the handler's first engine method call.

    Both paths must produce HTTP 503 with an OpenAI-shaped error body.
    """

    def test_chat_completions_ensure_engine_alive_race_returns_503(
        self, client, mock_engine_pool
    ):
        """Race window: get_engine succeeds, then ensure_engine_alive
        detects the enforcer evicted the engine before the handler
        could use it. Must surface as HTTP 503."""
        from omlx.exceptions import EngineEvictedError

        def raise_evicted(model_id, engine_ref):
            raise EngineEvictedError(model_id)

        mock_engine_pool.ensure_engine_alive = raise_evicted

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "evicted" in data["error"]["message"].lower()
        assert "retry" in data["error"]["message"].lower()


@pytest.mark.integration
class TestStoppedEngineReturns503:
    """Issue #4 server-level: the narrow race where a handler captured
    the engine reference, then ``BatchedEngine.stop()`` / non-streaming
    engine ``stop()`` completed between ``ensure_engine_alive`` and the
    handler invoking a method on the engine.

    Fixed path: the engine raises ``RequestAbortedError`` rather than a
    plain ``RuntimeError``, so ``request_aborted_handler`` in
    ``server.py`` translates it to HTTP 503 with an OpenAI-shaped error
    envelope instead of falling through to the generic 500 handler.
    """

    def test_chat_on_stopped_engine_returns_503(
        self, client, mock_llm_engine
    ):
        """Chat completions: stopped-engine error → HTTP 503."""
        from omlx.exceptions import RequestAbortedError

        mock_llm_engine.chat = AsyncMock(
            side_effect=RequestAbortedError(
                "Engine for test-model has been stopped "
                "due to memory pressure. Please retry the request."
            )
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "stopped" in data["error"]["message"].lower()
        assert "retry" in data["error"]["message"].lower()

    def test_completions_on_stopped_engine_returns_503(
        self, client, mock_llm_engine
    ):
        """Completions: stopped-engine error → HTTP 503."""
        from omlx.exceptions import RequestAbortedError

        mock_llm_engine.generate = AsyncMock(
            side_effect=RequestAbortedError(
                "Engine for test-model has been stopped "
                "due to memory pressure. Please retry the request."
            )
        )

        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Hello",
            },
        )

        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "stopped" in data["error"]["message"].lower()


@pytest.mark.integration
class TestBatchedEngineStoppedRaisesTypedError:
    """Issue #4 unit-level: BatchedEngine / VLMBatchedEngine must raise
    :class:`RequestAbortedError` (not plain ``RuntimeError``) when a
    method is called after ``stop()``. This is the underlying raise
    site that ``request_aborted_handler`` depends on.

    All four public async entry points — ``chat``, ``generate``,
    ``stream_chat``, ``stream_generate`` — must honour the contract.
    """

    @pytest.mark.asyncio
    async def test_chat_on_stopped_engine_raises_request_aborted_error(self):
        from omlx.engine.batched import BatchedEngine
        from omlx.exceptions import RequestAbortedError

        engine = BatchedEngine(model_name="test-model")
        engine._stopped = True
        engine._loaded = True

        with pytest.raises(RequestAbortedError) as exc_info:
            await engine.chat(
                messages=[{"role": "user", "content": "hi"}],
            )
        assert "test-model" in str(exc_info.value)
        assert "stopped" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_generate_on_stopped_engine_raises_request_aborted_error(
        self,
    ):
        from omlx.engine.batched import BatchedEngine
        from omlx.exceptions import RequestAbortedError

        engine = BatchedEngine(model_name="test-model")
        engine._stopped = True
        engine._loaded = True

        with pytest.raises(RequestAbortedError) as exc_info:
            await engine.generate(prompt="hello")
        assert "test-model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_generate_on_stopped_engine_raises_request_aborted_error(
        self,
    ):
        from omlx.engine.batched import BatchedEngine
        from omlx.exceptions import RequestAbortedError

        engine = BatchedEngine(model_name="test-model")
        engine._stopped = True
        engine._loaded = True

        with pytest.raises(RequestAbortedError) as exc_info:
            async for _ in engine.stream_generate(prompt="hello"):
                pass
        assert "test-model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_chat_on_stopped_engine_raises_request_aborted_error(
        self,
    ):
        from omlx.engine.batched import BatchedEngine
        from omlx.exceptions import RequestAbortedError

        engine = BatchedEngine(model_name="test-model")
        engine._stopped = True
        engine._loaded = True

        with pytest.raises(RequestAbortedError) as exc_info:
            async for _ in engine.stream_chat(
                messages=[{"role": "user", "content": "hi"}],
            ):
                pass
        assert "test-model" in str(exc_info.value)


@pytest.mark.integration
class TestBaseNonStreamingEngineStoppedRaisesTypedError:
    """Issue #4 for non-streaming engines: ``embed``, ``rerank``,
    ``transcribe``, ``synthesize``, ``process`` must raise
    :class:`RequestAbortedError` (not plain ``RuntimeError``) when the
    engine has been stopped, because ``stop()`` sets the cooperative
    abort flag and the ``_raise_if_aborted`` check is now ordered
    before the ``_model is None`` guard. A handler racing with stop
    therefore receives HTTP 503 instead of 500.
    """

    @pytest.mark.asyncio
    async def test_embed_after_stop_raises_request_aborted_error(self):
        from omlx.engine.embedding import EmbeddingEngine
        from omlx.exceptions import RequestAbortedError
        from unittest.mock import MagicMock, patch

        with patch("omlx.engine.embedding.MLXEmbeddingModel"):
            engine = EmbeddingEngine("fake-embed-model")
            # Start with a model attached, then stop it. stop() must
            # set the abort flag before clearing self._model.
            engine._model = MagicMock()
            await engine.stop()

            with pytest.raises(RequestAbortedError) as exc_info:
                await engine.embed(["hello"])
            assert "fake-embed-model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rerank_after_stop_raises_request_aborted_error(self):
        from omlx.engine.reranker import RerankerEngine
        from omlx.exceptions import RequestAbortedError
        from unittest.mock import MagicMock, patch

        with patch("omlx.engine.reranker.MLXRerankerModel"):
            engine = RerankerEngine("fake-rerank-model")
            engine._model = MagicMock()
            await engine.stop()

            with pytest.raises(RequestAbortedError) as exc_info:
                await engine.rerank(query="q", documents=["a"])
            assert "fake-rerank-model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_embed_still_raises_runtime_error_when_never_started(self):
        """When the engine was never started at all (programming error,
        not a race), the distinct RuntimeError("Engine not started")
        still fires — the _aborted flag is clean, so _raise_if_aborted
        is a no-op and the model guard runs.
        """
        from omlx.engine.embedding import EmbeddingEngine

        engine = EmbeddingEngine("fake-embed-model")
        # Never started — _model is None, _aborted is not set.
        with pytest.raises(RuntimeError, match="Engine not started"):
            await engine.embed(["hello"])


@pytest.mark.integration
class TestStreamingAbortSurfacesCleanSSEError:
    """Issue #5 (fixed): when a stream is aborted mid-flight, the
    client sees a clean SSE error event — never the error text
    embedded as assistant content, and never a non-standard
    ``finish_reason``.

    Option A fix: ``engine_core.stream_outputs`` raises
    :class:`RequestAbortedError` on error outputs instead of yielding
    them and breaking. The streaming handlers already wrap their
    ``async for`` loops in ``except Exception`` blocks that emit a
    proper SSE error chunk + ``[DONE]`` and return. This is simulated
    here by making the mock ``stream_chat`` / ``stream_generate``
    raise the same typed exception the real engine would raise.

    Also verifies the public/private error split: the client-facing
    message must not contain operator-only hints like the
    ``--max-process-memory`` CLI flag name.
    """

    def _parse_sse(self, body: str) -> list[dict]:
        """Parse ``data: {...}`` SSE lines into parsed JSON dicts."""
        import json

        out: list[dict] = []
        for line in body.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]" or not payload:
                continue
            try:
                out.append(json.loads(payload))
            except json.JSONDecodeError:
                continue
        return out

    # Client-facing error message matching what
    # engine_core.abort_all_requests puts into the error RequestOutput.
    # Intentionally excludes operator-only hints like flag names.
    PUBLIC_ABORT_MSG = (
        "Request aborted due to server memory pressure. "
        "Please retry the request."
    )

    def _chat_stream(self, client, mock_llm_engine):
        from omlx.exceptions import RequestAbortedError

        async def abort_stream(messages, **kwargs):
            # Mirror the real stream_outputs path: raise BEFORE
            # yielding anything. No error text leaks as content.
            raise RequestAbortedError(self.PUBLIC_ABORT_MSG)
            yield  # pragma: no cover — unreachable, keeps it a generator

        mock_llm_engine.stream_chat = abort_stream
        return client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

    def _completion_stream(self, client, mock_llm_engine):
        from omlx.exceptions import RequestAbortedError

        async def abort_stream(prompt, **kwargs):
            raise RequestAbortedError(self.PUBLIC_ABORT_MSG)
            yield  # pragma: no cover

        mock_llm_engine.stream_generate = abort_stream
        return client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Hello",
                "stream": True,
            },
        )

    def test_chat_stream_aborted_has_no_content_leak(
        self, client, mock_llm_engine
    ):
        """No chat content delta carries the abort message."""
        response = self._chat_stream(client, mock_llm_engine)
        assert response.status_code == 200
        chunks = self._parse_sse(response.text)

        for c in chunks:
            if "error" in c:
                continue
            for choice in c.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                assert "Request aborted" not in content, (
                    f"abort message leaked into content delta: {content!r}"
                )
                assert "memory" not in content.lower(), (
                    f"abort-related text leaked into content delta: "
                    f"{content!r}"
                )

    def test_chat_stream_aborted_emits_clean_sse_error_chunk(
        self, client, mock_llm_engine
    ):
        """A dedicated SSE error chunk carries the public abort message,
        followed by a ``[DONE]`` terminator. This is the path produced
        by the streaming handler's ``except Exception:`` block.
        """
        response = self._chat_stream(client, mock_llm_engine)
        assert response.status_code == 200
        assert response.text.rstrip().endswith("data: [DONE]")

        chunks = self._parse_sse(response.text)
        error_chunks = [c for c in chunks if "error" in c]
        assert len(error_chunks) == 1, (
            f"Expected exactly one SSE error chunk, got "
            f"{len(error_chunks)}. chunks={chunks}"
        )
        err = error_chunks[0]["error"]
        # Public message is present, operator hints are not.
        assert "Request aborted" in err["message"]
        assert "retry" in err["message"].lower()

    def test_chat_stream_aborted_public_error_has_no_operator_leaks(
        self, client, mock_llm_engine
    ):
        """The public-facing abort message must not contain operator-only
        hints (CLI flag names, ``--max-process-memory``, internal
        variable names)."""
        response = self._chat_stream(client, mock_llm_engine)
        body = response.text
        assert "--max-process-memory" not in body, (
            "Operator-only CLI flag name leaked to client"
        )
        assert "_max_bytes" not in body

    def test_chat_stream_aborted_no_nonstandard_finish_reason(
        self, client, mock_llm_engine
    ):
        """The final chunk must NOT carry ``finish_reason="error"``.
        With the fix, the handler returns early via its exception path
        and never emits a final content-chunk finish_reason at all.
        """
        response = self._chat_stream(client, mock_llm_engine)
        chunks = self._parse_sse(response.text)

        valid = {"stop", "length", "tool_calls", "content_filter", None}
        for c in chunks:
            if "error" in c:
                continue
            for choice in c.get("choices", []):
                fr = choice.get("finish_reason")
                assert fr in valid, (
                    f"Non-standard finish_reason={fr!r} leaked to "
                    f"client in chunk: {c}"
                )

    def test_completion_stream_aborted_emits_clean_sse_error_chunk(
        self, client, mock_llm_engine
    ):
        """Same invariants apply to ``/v1/completions`` streaming via
        ``stream_completion`` — both handlers share the same
        ``except Exception`` pattern."""
        response = self._completion_stream(client, mock_llm_engine)
        assert response.status_code == 200
        assert response.text.rstrip().endswith("data: [DONE]")

        chunks = self._parse_sse(response.text)
        error_chunks = [c for c in chunks if "error" in c]
        assert len(error_chunks) == 1
        # No content-shaped chunk contains the abort message text.
        for c in chunks:
            if "error" in c:
                continue
            for choice in c.get("choices", []):
                text = choice.get("text") or ""
                assert "Request aborted" not in text, (
                    f"abort message leaked into completion text: {text!r}"
                )