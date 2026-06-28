# PR title

fix(crescendo): isolate conversation memory per attack to prevent cross-vulnerability prompt bleed

# Summary

`CrescendoJailbreaking` keeps its per-conversation state — a `MemorySystem` plus
`red_teaming_chat_conversation_id` / `target_conversation_id` — as **instance
attributes set once in `__init__`**. But a single attack instance is reused across
every vulnerability that samples it (`AttackSimulator` selects it via
`random.choices`), and those conversations run **concurrently** under
`asyncio.gather`. They all append to that one shared buffer, so each round's
attacker prompt (`json.dumps(red_teaming_history)`) accumulates **other
vulnerabilities' turns**.

### Impact

- **Degraded attacks:** the attacker model receives several different
  "generate the next question for Vulnerability X" instructions in a single prompt
  and emits one question — the multi-turn escalation context is corrupted.
- **Token blow-up:** every attacker prompt carries all concurrent conversations'
  histories (roughly quadratic in the number of concurrent Crescendo conversations).

### Why only Crescendo

The other multi-turn attacks (Linear, Tree, Sequential, BadLikertJudge) keep
conversation history in a method-local `turns` list (or per-traversal nodes), so
concurrent calls on a shared instance never collide. Crescendo is the only one that
hoisted that state onto the instance.

### Fix

Move the per-conversation state off the instance into a `CrescendoConversation`
created fresh in each `_get_turns` / `_a_get_turns` call and threaded through the
helper methods — the same stack-local pattern the other attacks already use (and
the `BatchContext` dataclass in `trace_scanner.py`). Note: `max_concurrent=1` is
*not* a fix, because there is no per-call reset, so sequential reuse of one instance
accumulates history too.

### Testing

- Added `test_no_cross_conversation_bleed_on_shared_instance`: runs three
  concurrent conversations through one instance with a recording simulator and
  asserts no attacker prompt mixes vulnerabilities — fails on the old design,
  passes now.
- Updated the existing Crescendo tests for the per-call state; all pass.
