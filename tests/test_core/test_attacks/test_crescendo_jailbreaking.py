import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock
from deepeval.models import DeepEvalBaseLLM
from deepteam.attacks.multi_turn import CrescendoJailbreaking
from deepteam.attacks.multi_turn.crescendo_jailbreaking.crescendo_jailbreaking import (
    MemorySystem,
    CrescendoConversation,
)
from deepteam.attacks.multi_turn.crescendo_jailbreaking.schema import (
    AttackData,
    RefusalData,
    EvalData,
)
from deepteam.vulnerabilities import Bias
from deepteam.test_case.test_case import RTTurn


class TestCrescendoJailbreaking:

    BIAS = Bias(types=["race"])

    def test_crescendo_jailbreaking_initialization(self):
        attack = CrescendoJailbreaking()
        assert attack.weight == 1
        assert attack.max_rounds == 10
        assert attack.max_backtracks == 10
        assert attack.get_name() == "Crescendo Jailbreaking"

    def test_crescendo_jailbreaking_initialization_with_params(self):
        attack = CrescendoJailbreaking(weight=3, max_rounds=5, max_backtracks=3)
        assert attack.weight == 3
        assert attack.max_rounds == 5
        assert attack.max_backtracks == 3
        assert attack.get_name() == "Crescendo Jailbreaking"

    def test_memory_system_class(self):
        memory = MemorySystem()
        cid = "c1"
        memory.add_message(cid, {"role": "user", "content": "a"})
        memory.add_message(cid, {"role": "assistant", "content": "b"})
        assert len(memory.get_conversation(cid)) == 2
        new_cid = memory.duplicate_conversation_excluding_last_turn(cid)
        assert new_cid != cid
        # the duplicate drops the last user/assistant pair
        assert memory.get_conversation(new_cid) == []

    def test_per_conversation_state_is_not_on_the_instance(self):
        # Per-conversation memory/ids must NOT live on the shared instance — that is
        # exactly what let concurrent attacks bleed into one another. They belong to
        # the per-call CrescendoConversation instead.
        attack = CrescendoJailbreaking()
        assert not hasattr(attack, "memory")
        assert not hasattr(attack, "target_conversation_id")
        assert not hasattr(attack, "red_teaming_chat_conversation_id")

    def test_crescendo_jailbreaking_enhance_and_turns(self):
        attack = CrescendoJailbreaking()
        mock_callback = MagicMock(return_value="Mock response")
        assistant_only_turns = [
            RTTurn(role="assistant", content="Assistant content")
        ]
        with pytest.raises(ValueError):
            attack._get_turns(mock_callback, assistant_only_turns)

    @pytest.mark.asyncio
    async def test_crescendo_jailbreaking_async_enhance_and_turns(self):
        attack = CrescendoJailbreaking()
        mock_callback = AsyncMock(return_value="Mock response")
        assistant_only_turns = [
            RTTurn(role="assistant", content="Assistant content")
        ]
        with pytest.raises(ValueError):
            await attack._a_get_turns(mock_callback, assistant_only_turns)

    def test_crescendo_jailbreaking_backtrack_memory(self):
        attack = CrescendoJailbreaking()
        state = CrescendoConversation(simulator_model=None, model_callback=None)
        state.memory.add_message(
            state.target_conversation_id, {"role": "user", "content": "a"}
        )
        state.memory.add_message(
            state.target_conversation_id, {"role": "assistant", "content": "b"}
        )

        new_conv_id = attack.backtrack_memory(
            state, state.target_conversation_id
        )

        # Should return a different conversation ID
        assert new_conv_id != state.target_conversation_id
        assert isinstance(new_conv_id, str)

    @pytest.mark.asyncio
    async def test_no_cross_conversation_bleed_on_shared_instance(self):
        """Regression: one CrescendoJailbreaking instance, reused across vulnerabilities
        running concurrently, must keep each conversation's attacker prompt isolated.

        Before the per-call CrescendoConversation, all conversations shared
        self.memory under one red_teaming_chat_conversation_id, so each attacker
        prompt accumulated *other* vulnerabilities' turns.
        """
        vulns = [
            ("Bias", "race"),
            ("Toxicity", "insults"),
            ("Misinformation", "factual_errors"),
        ]

        class RecordingSim(DeepEvalBaseLLM):
            def __init__(self):
                self.model_name = "recording-sim"
                self.attack_prompts = []

            def load_model(self):
                return self

            def get_model_name(self):
                return self.model_name

            def _respond(self, prompt, schema):
                if schema is AttackData:
                    self.attack_prompts.append(prompt)
                    return AttackData(
                        generated_question="Q",
                        last_response_summary="S",
                        rationale_behind_jailbreak="R",
                    )
                if schema is RefusalData:
                    return RefusalData(value=False, rationale="no", metadata=0)
                if schema is EvalData:
                    return EvalData(
                        value=False, description="d", rationale="r", metadata=0
                    )
                raise AssertionError(f"unexpected schema {schema}")

            def generate(self, prompt, schema=None):
                return self._respond(prompt, schema)

            async def a_generate(self, prompt, schema=None):
                # let concurrent conversations interleave at the await point
                await asyncio.sleep(0)
                return self._respond(prompt, schema)

        async def target(content, turns=None):
            await asyncio.sleep(0)
            return RTTurn(role="assistant", content="benign target response")

        sim = RecordingSim()
        # ONE instance, reused across all three concurrent conversations
        attack = CrescendoJailbreaking(max_rounds=3, simulator_model=sim)

        async def drive(name, vtype):
            return await attack._a_get_turns(
                model_callback=target,
                turns=[RTTurn(role="user", content=f"seed for {name}")],
                vulnerability=name,
                vulnerability_type=vtype,
                simulator_model=sim,
            )

        await asyncio.gather(*[drive(n, t) for n, t in vulns])

        assert sim.attack_prompts, "expected attacker prompts to be captured"
        names = [n for n, _ in vulns]
        for prompt in sim.attack_prompts:
            present = {n for n in names if f"Vulnerability: {n} |" in prompt}
            assert (
                len(present) <= 1
            ), f"attacker prompt leaked multiple vulnerabilities: {sorted(present)}"

    def test_crescendo_jailbreaking_has_required_methods(self):
        attack = CrescendoJailbreaking()

        # Verify all required methods exist
        assert hasattr(attack, "progress")
        assert hasattr(attack, "a_progress")
        assert hasattr(attack, "get_name")
        assert hasattr(attack, "_get_turns")
        assert hasattr(attack, "_a_get_turns")
        assert callable(attack.progress)
        assert callable(attack.a_progress)
        assert callable(attack.get_name)
        assert callable(attack._get_turns)
        assert callable(attack._a_get_turns)
