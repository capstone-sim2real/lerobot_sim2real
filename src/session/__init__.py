"""Arm session layer shared by the rule-based runners and the LLM agent.

Nothing here is imported by ``import session``: every submodule is imported
explicitly, so pulling in the cancellation token never pulls in the IK or
camera stack (AGENTS.md §2).
"""
