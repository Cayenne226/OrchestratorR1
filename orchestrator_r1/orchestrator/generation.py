from dataclasses import dataclass, field
from typing import Optional, List, Dict
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch

from .parser import parse_output, extract_answer, STOP_TOKENS
from .context_manager import compress_context, CompressionConfig
from ..agent_pool.agent_registry import AgentRegistry
from ..prompts.system_prompt import SYSTEM_PROMPT


@dataclass
class GenerationConfig:
    max_turns: int = 6
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    max_obs_length: int = 800           # max chars of agent response injected per turn
    max_prompt_length: int = 4096        # max tokens for the orchestrator's input window
    enable_context_compression: bool = True
    compression_budget_ratio: float = 0.80
    keep_recent_turns: int = 2
    middle_block_max_chars: int = 200


@dataclass
class RolloutResult:
    full_text: str                          # complete context (prompt + all turns)
    answer: Optional[str]                   # extracted final answer
    agent_calls: List[dict] = field(default_factory=list)
    n_turns: int = 0
    total_cost: float = 0.0
    token_ids: List[int] = field(default_factory=list)
    n_compressions: int = 0


class OrchestratorGenerationManager:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        agent_registry: AgentRegistry,
        config: Optional[GenerationConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.registry = agent_registry
        self.config = config or GenerationConfig()
        self._compression_cfg = CompressionConfig(
            max_context_tokens=self.config.max_prompt_length,
            budget_ratio=self.config.compression_budget_ratio,
            keep_recent_turns=self.config.keep_recent_turns,
            middle_block_max_chars=self.config.middle_block_max_chars,
        )

        # Token IDs for stopping conditions
        self._stop_ids = self._build_stop_ids()

    def _build_stop_ids(self) -> Dict[str, List[int]]:
        stop_ids = {}
        for token in STOP_TOKENS:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            stop_ids[token] = ids
        return stop_ids

    def _build_prompt(self, user_input: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_input},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _compress_if_needed(self, context: str) -> tuple[str, dict]:
        """Apply context compression if enabled and over budget."""
        if not self.config.enable_context_compression:
            return context, {"compressed": False}
        return compress_context(
            context,
            config=self._compression_cfg,
            tokenizer=self.tokenizer,
        )

    @torch.no_grad()
    def _generate_step(self, context: str) -> str:
        """Generate until </call> or </answer> stop token."""
        inputs = self.tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode only the new tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def rollout(self, user_input: str) -> RolloutResult:
        """Run one complete orchestration rollout for a single input."""
        context = self._build_prompt(user_input)
        agent_calls = []
        total_cost = 0.0
        n_compressions = 0

        for turn in range(self.config.max_turns):
            # Generate next segment
            generated = self._generate_step(context)
            context += generated

            # Parse what the model produced
            parsed = parse_output(generated)

            # If model wants to call an agent
            if parsed.call is not None:
                call = parsed.call
                response, cost = self.registry.dispatch(
                    call.agent_type, call.query, tier=call.tier
                )
                total_cost += cost
                agent_calls.append({
                    "agent_type": call.agent_type,
                    "tier": call.tier,
                    "query": call.query,
                    "cost": cost,
                    "turn": turn,
                })
                # Truncate long responses (per-turn cap)
                if len(response) > self.config.max_obs_length:
                    response = response[:self.config.max_obs_length] + "..."
                # Inject agent response into context
                context += f"\n<information>{response}</information>\n"

                # Context budget check: compress middle blocks if over budget
                context, comp_meta = self._compress_if_needed(context)
                if comp_meta.get("compressed"):
                    n_compressions += 1
                continue

            # If model produced a final answer
            if parsed.answer is not None:
                return RolloutResult(
                    full_text=context,
                    answer=parsed.answer,
                    agent_calls=agent_calls,
                    n_turns=turn + 1,
                    total_cost=total_cost,
                    n_compressions=n_compressions,
                )

        # Max turns reached without <answer>
        answer = extract_answer(context)
        return RolloutResult(
            full_text=context,
            answer=answer,
            agent_calls=agent_calls,
            n_turns=self.config.max_turns,
            total_cost=total_cost,
            n_compressions=n_compressions,
        )

    def rollout_batch(self, inputs: List[str]) -> List[RolloutResult]:
        """Run rollouts for a batch of inputs (sequential for now)."""
        return [self.rollout(inp) for inp in inputs]
