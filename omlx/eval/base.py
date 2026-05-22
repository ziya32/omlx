# SPDX-License-Identifier: Apache-2.0
"""Base classes and data models for accuracy benchmarks."""

import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Token budget for thinking/reasoning models (industry reference: OpenCompass 8K~32K)
THINKING_MIN_TOKENS = 8192
THINKING_MAX_TOKENS = 32768


@dataclass
class QuestionResult:
    """Result for a single benchmark question."""

    question_id: str
    correct: bool
    expected: str
    predicted: str
    time_seconds: float
    question_text: str = ""
    raw_response: str = ""
    category: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Aggregated result for a complete benchmark run."""

    benchmark_name: str
    accuracy: float
    total_questions: int
    correct_count: int
    time_seconds: float
    question_results: list[QuestionResult] = field(default_factory=list)
    category_scores: Optional[dict[str, float]] = None
    thinking_used: bool = False


class BaseBenchmark(ABC):
    """Abstract base class for accuracy benchmarks."""

    name: str = ""
    quick_size: int = 100
    # Max concurrent answer-scoring operations. 0 ties it to the generation
    # concurrency (batch_size). Benchmarks whose check_answer is a cheap
    # string/number comparison leave this at 0; code benchmarks that execute
    # generated code in subprocesses set a fixed cap so scoring can neither
    # storm CPU/memory nor stall generation, no matter how high batch_size is.
    score_concurrency: int = 0

    @abstractmethod
    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load dataset items.

        Args:
            sample_size: Number of questions to sample. 0 = full dataset.

        Returns:
            List of dataset items (format varies by benchmark).
        """
        pass

    @abstractmethod
    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format a dataset item into chat messages for the engine.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        pass

    @abstractmethod
    def extract_answer(self, response: str, item: dict) -> str:
        """Extract the predicted answer from model response text."""
        pass

    @abstractmethod
    def check_answer(self, predicted: str, item: dict) -> bool:
        """Check if the predicted answer is correct."""
        pass

    def get_max_tokens(self) -> int:
        """Max tokens to generate per question. Override for longer answers."""
        return 128

    def get_category(self, item: dict) -> Optional[str]:
        """Return category/subject for per-category scoring. None if N/A."""
        return None

    def get_question_text(self, item: dict) -> str:
        """Return a human-readable question text for result export."""
        return item.get("question", item.get("description", item.get("context", "")))

    @staticmethod
    def _extract_mc_answer(response: str, valid_letters: list[str]) -> str:
        """Extract multiple choice answer from response.

        Strategy:
        1. Look for explicit "answer is X" / "answer: X" patterns (last match)
        2. Fall back to last valid letter in response
        3. Case-insensitive
        """
        response_upper = response.strip().upper()
        pattern_letters = "".join(valid_letters)

        # 1. Look for "answer is X", "answer: X", "answer X" patterns — use LAST match
        answer_patterns = re.findall(
            r"(?:answer\s*(?:is|:)\s*)([" + pattern_letters + r"])\b",
            response_upper,
        )
        if answer_patterns:
            return answer_patterns[-1]

        # 2. Fall back to last valid letter with word boundary
        all_matches = re.findall(
            r"\b([" + pattern_letters + r"])\b",
            response_upper,
        )
        if all_matches:
            return all_matches[-1]

        # 3. Check first character
        if response.strip() and response.strip()[0].upper() in valid_letters:
            return response.strip()[0].upper()

        return ""

    @staticmethod
    def _extract_last_code_block(response: str) -> str:
        """Extract the LAST code block from model response.

        Uses last match to avoid picking up drafts/examples.
        Falls back to line-by-line detection if no code blocks found.
        """
        response = response.strip()

        # Find ALL python code blocks, use LAST
        blocks = re.findall(r"```python\s*\n(.*?)```", response, re.DOTALL)
        if blocks:
            return blocks[-1].strip()

        # Generic code blocks
        blocks = re.findall(r"```\s*\n(.*?)```", response, re.DOTALL)
        if blocks:
            return blocks[-1].strip()

        # Line-by-line fallback
        lines = response.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            if not in_code and (
                line.startswith("def ")
                or line.startswith("class ")
                or line.startswith("import ")
                or line.startswith("from ")
                or line.startswith("#")
            ):
                in_code = True
            if in_code:
                code_lines.append(line)

        return "\n".join(code_lines) if code_lines else response

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Remove <think>...</think> blocks from model output."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    async def _eval_single(
        self, engine: Any, item: dict, index: int,
        sampling_kwargs: Optional[dict] = None,
        enable_thinking: bool = False,
    ) -> tuple[int, dict, str, str, str]:
        """Evaluate a single item.

        Returns (index, item, response_text, prompt_text, raw_text).
        raw_text is the unstripped output for auto-detection of thinking tags.
        """
        messages = self.format_prompt(item)
        prompt_text = "\n".join(m.get("content", "") for m in messages)
        kwargs = dict(sampling_kwargs or {})
        # Force benchmark-controlled params (override model settings)
        max_tokens = self.get_max_tokens()
        # Harmony models (gpt_oss) use analysis + final channels;
        # analysis can consume the entire budget before final is emitted
        if getattr(engine, "model_type", None) == "gpt_oss":
            max_tokens = max(max_tokens * 4, 8192)
        elif enable_thinking:
            max_tokens = min(
                max(max_tokens, THINKING_MIN_TOKENS), THINKING_MAX_TOKENS
            )
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = 0.0
        kwargs["presence_penalty"] = 0.0
        kwargs["repetition_penalty"] = 1.0
        # Merge enable_thinking into any existing chat_template_kwargs
        ct_kwargs = kwargs.pop("chat_template_kwargs", {}) or {}
        ct_kwargs["enable_thinking"] = enable_thinking
        kwargs["chat_template_kwargs"] = ct_kwargs
        try:
            output = await engine.chat(
                messages=messages,
                **kwargs,
            )
            raw_text = output.text
            text = self._strip_think_tags(raw_text)
            return index, item, text, prompt_text, raw_text
        except Exception as e:
            logger.warning(f"Engine error on question {index}: {e}")
            return index, item, "", prompt_text, ""

    async def _score_result(
        self, item: dict, response_text: str
    ) -> tuple[str, bool]:
        """Extract and check the answer for one generated response.

        Returns (predicted, is_correct). The default runs inline, which is
        correct for benchmarks whose check_answer is a cheap string/number
        comparison. Code benchmarks override this to offload subprocess
        execution to a thread so it doesn't block in-flight generation.
        """
        predicted = self.extract_answer(response_text, item)
        is_correct = self.check_answer(predicted, item)
        return predicted, is_correct

    def _expected_label(self, item: dict) -> str:
        """Value stored as the 'expected' answer in exported results."""
        return str(item.get("answer", ""))

    def _predicted_for_export(self, predicted: str) -> str:
        """Transform the predicted answer for storage/export (e.g. truncate)."""
        return predicted

    async def run(
        self,
        engine: Any,
        items: list[dict],
        on_progress: Optional[Callable[[int, int], Any]] = None,
        batch_size: int = 1,
        sampling_kwargs: Optional[dict] = None,
        enable_thinking: bool = False,
    ) -> BenchmarkResult:
        """Run the benchmark on all items.

        Generation runs as a sliding window that keeps ``batch_size`` requests
        in flight at all times: the moment one response returns, the next item
        is dispatched, rather than waiting for a whole batch to finish. Answer
        scoring happens as each response arrives, bounded separately (see
        ``score_concurrency``) so a heavy scorer can't stall generation.

        Args:
            engine: oMLX engine instance with chat() method.
            items: Dataset items to evaluate.
            on_progress: Callback(current, total) for progress reporting.
            batch_size: Number of concurrent requests in flight (1 = sequential).
            enable_thinking: Enable thinking mode for reasoning models. When
                False, probes the first item and, if the model emits <think>
                tags anyway, switches the whole run to thinking mode (raising
                the token budget so the answer isn't starved by the think block).

        Returns:
            BenchmarkResult with accuracy and per-question details.
        """
        start_time = time.time()
        total = len(items)
        if total == 0:
            return BenchmarkResult(
                benchmark_name=self.name,
                accuracy=0.0,
                total_questions=0,
                correct_count=0,
                time_seconds=0.0,
                question_results=[],
                category_scores=None,
                thinking_used=enable_thinking,
            )

        concurrency = max(1, batch_size)
        thinking_used = enable_thinking

        # Probe item 0 to detect models that emit <think> despite
        # enable_thinking=False. The default answer budget is small, so a think
        # block would consume it before the answer appears; detecting this lets
        # the whole run use the larger thinking budget instead. The probe's
        # output is reused (not regenerated) when no switch is needed.
        probe: Optional[tuple] = None
        if not thinking_used:
            t0 = time.perf_counter()
            probe_out = await self._eval_single(
                engine, items[0], 0, sampling_kwargs, False
            )
            probe_latency = time.perf_counter() - t0
            if "<think>" in probe_out[4]:
                logger.warning(
                    f"{self.name}: model outputs <think> tags with "
                    "enable_thinking=False, switching to thinking mode"
                )
                thinking_used = True
            else:
                probe = (probe_out, probe_latency)

        gen_sem = asyncio.Semaphore(concurrency)
        score_sem = asyncio.Semaphore(
            self.score_concurrency if self.score_concurrency > 0 else concurrency
        )
        # index -> (item, predicted, is_correct, response_text, prompt_text, latency)
        collected: dict[int, tuple] = {}
        completed = 0
        progress_lock = asyncio.Lock()

        async def handle(
            index: int, item: dict, pre: Optional[tuple] = None
        ) -> None:
            nonlocal completed
            if pre is None:
                async with gen_sem:
                    t = time.perf_counter()
                    out = await self._eval_single(
                        engine, item, index, sampling_kwargs, thinking_used
                    )
                    latency = time.perf_counter() - t
            else:
                out, latency = pre
            response_text, prompt_text = out[2], out[3]
            async with score_sem:
                predicted, is_correct = await self._score_result(item, response_text)
            collected[index] = (
                item, predicted, is_correct, response_text, prompt_text, latency
            )
            async with progress_lock:
                completed += 1
                current = completed
            if on_progress:
                await on_progress(current, total)

        tasks = [
            asyncio.create_task(
                handle(i, item, probe if (i == 0 and probe is not None) else None)
            )
            for i, item in enumerate(items)
        ]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            # Includes CancelledError (raised by on_progress on user cancel):
            # cancel any still-running generations before propagating.
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        # Aggregate in item order.
        correct = 0
        category_correct: dict[str, int] = {}
        category_total: dict[str, int] = {}
        results: list[QuestionResult] = []
        for index in range(total):
            item, predicted, is_correct, response_text, prompt_text, latency = (
                collected[index]
            )
            if is_correct:
                correct += 1

            cat = self.get_category(item)
            if cat is not None:
                category_total[cat] = category_total.get(cat, 0) + 1
                if is_correct:
                    category_correct[cat] = category_correct.get(cat, 0) + 1

            results.append(
                QuestionResult(
                    question_id=str(item.get("id", index)),
                    correct=is_correct,
                    expected=self._expected_label(item),
                    predicted=self._predicted_for_export(predicted),
                    time_seconds=latency,
                    question_text=prompt_text,
                    raw_response=response_text,
                    category=cat,
                )
            )

        total_time = time.time() - start_time
        accuracy = correct / total if total > 0 else 0.0

        cat_scores = None
        if category_total:
            cat_scores = {}
            for cat in sorted(category_total.keys()):
                cat_scores[cat] = (
                    category_correct.get(cat, 0) / category_total[cat]
                    if category_total[cat] > 0
                    else 0.0
                )

        return BenchmarkResult(
            benchmark_name=self.name,
            accuracy=accuracy,
            total_questions=total,
            correct_count=correct,
            time_seconds=total_time,
            question_results=results,
            category_scores=cat_scores,
            thinking_used=thinking_used,
        )
