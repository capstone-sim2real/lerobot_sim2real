"""SO-101 LLM tool-calling agent (so101-agent).

Import-light by design: no LLM SDK, web framework, lerobot or placo is
imported by ``import agent`` or by the provider-neutral modules
(``agent.provider.types``, ``agent.provider.schema``, ``agent.tools``,
``agent.runner``, ``agent.control``). SDKs load inside their adapter's
constructor (AGENTS.md §2).
"""
