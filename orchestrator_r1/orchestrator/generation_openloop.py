"""
Ablation variant: w/o reactive (open-loop orchestration).

Instead of injecting <information> blocks after each agent call and letting the
model react, this variant:
  1. Lets the model generate ALL <call> tags at once (no intermediate feedback)
  2. Executes all calls in batch
  3. Appends all results at the end
  4. Model then generates <answer> based on bulk information

This simulates what Conductor does: plan-then-execute without intermediate feedback.
It is the MOST IMPORTANT ablation — it directly tests the reactive vs open-loop claim.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from transformers import PreTrainedModel, PreTrainedTokenizer
import re
import torch

from .parser import parse_output, extract_answer, STOP_TOKENS, VALID_AGENT_TYPES
from ..agent_pool.agent_registry import AgentRegistry
from ..prompts.system_prompt import SYSTEM_PROMPT
from .generation import GenerationConfig, RolloutResult


class OpenLoopGenerationManager:
    """Open-loop orchestration: generate all calls first, execute, then answer.

    Differences from OrchestratorGenerationManager:
    - Phase 1: Model generates ALL <call> tags without receiving any <information>
    - Phase 2: All calls are executed in batch (parallel)
    - Phase 3: All results are injected as a single block
    - Phase 4: Model generates final <answer>
    """

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

    def _build_prompt(self, user_input: str) -> str:
        # Same system prompt but with modified instruction
        modified_prompt = SYSTEM_PROMPT + (
            "\n\nIMPORTANT: Plan all the agent calls you need upfront. "
            "Generate ALL <call> tags in sequence before receiving any results. "
            "You will receive all results at once after planning."
        )
        messages = [
            {"role": "system", "content": modified_prompt},
            {"role": "user", "content": user_input},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    @torch.no_grad()
    def _generate_step(self, context: str, max_new_tokens: int = None) -> str:
        """Generate until </answer> or max tokens."""
        inputs = self.tokenizer(
            context, return_tensors="pt", truncation=True, max_length=4096,
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def _extract_all_calls(self, text: str) -> List[dict]:
        """Extract all <call type="X" [tier="Y"]>query</call> from generated text."""
        calls = []
        for m in re.finditer(r'<call\s+type="(\w+)"([^>]*)>(.*?)</call>', text, re.DOTALL):
            agent_type = m.group(1).strip()
            attrs = m.group(2)
            query = m.group(3).strip()
            if agent_type in VALID_AGENT_TYPES:
                call = {"agent_type": agent_type, "query": query}
                if agent_type == "executor":
                    tier_match = re.search(r'tier="(\w+)"', attrs)
                    call["tier"] = (
                        tier_match.group(1) if tier_match and tier_match.group(1) in {"strong", "weak"}
                        else "weak"
                    )
                calls.append(call)
        return calls

    def rollout(self, user_input: str) -> RolloutResult:
        """Open-loop rollout:
        1. Generate plan (all calls at once)
        2. Execute all calls in batch
        3. Inject all results
        4. Generate answer
        """
        context = self._build_prompt(user_input)

        # Phase 1: Generate plan — let model output multiple <call> tags
        # Give it more tokens since it needs to plan everything upfront
        plan_text = self._generate_step(context, max_new_tokens=self.config.max_new_tokens * 2)
        context += plan_text

        # Check if model already gave an answer (trivial case)
        answer = extract_answer(plan_text)
        if answer is not None:
            return RolloutResult(
                full_text=context, answer=answer,
                agent_calls=[], n_turns=1, total_cost=0.0,
            )

        # Phase 2: Extract and execute all planned calls
        planned_calls = self._extract_all_calls(plan_text)

        if not planned_calls:
            # No calls generated — treat as direct answer attempt
            return RolloutResult(
                full_text=context, answer=extract_answer(context),
                agent_calls=[], n_turns=1, total_cost=0.0,
            )

        # Execute all calls in parallel
        results_list = self.registry.dispatch_batch(planned_calls)

        agent_calls = []
        total_cost = 0.0
        all_info_blocks = []

        for call, (response, cost) in zip(planned_calls, results_list):
            total_cost += cost
            agent_calls.append({
                "agent_type": call["agent_type"],
                "tier": call.get("tier"),
                "query": call["query"],
                "cost": cost,
                "turn": 0,  # all in "turn 0" since it's open-loop
            })
            # Truncate long responses
            if len(response) > self.config.max_obs_length:
                response = response[:self.config.max_obs_length] + "..."
            all_info_blocks.append(
                f"<information source=\"{call['agent_type']}\">{response}</information>"
            )

        # Phase 3: Inject all results at once
        context += "\n\n" + "\n".join(all_info_blocks) + "\n\n"
        context += "Based on all the information above, provide your final answer.\n"

        # Phase 4: Generate final answer
        answer_text = self._generate_step(context, max_new_tokens=self.config.max_new_tokens)
        context += answer_text

        answer = extract_answer(answer_text) or extract_answer(context)

        return RolloutResult(
            full_text=context,
            answer=answer,
            agent_calls=agent_calls,
            n_turns=1,  # open-loop = 1 "turn" of planning + 1 "turn" of answering
            total_cost=total_cost,
        )

    def rollout_batch(self, inputs: List[str]) -> List[RolloutResult]:
        return [self.rollout(inp) for inp in inputs]
