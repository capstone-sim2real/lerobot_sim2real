"""Idle ticks, recorded tool sequences and dataset episodes."""
from __future__ import annotations

import time



class EpisodeMixin:
    def idle_tick(self):
        self.collection.idle_tick()

    @property
    def idle_poll_s(self):
        return self.cfg.agent.collection.idle_poll_s if self.collection.recording else None

    def end_command(self):
        # Never leave background recording running after the owning chat/direct command.
        self.collection.discard("turn_ended")

    def on_tool_result(self, action, result):
        self.collection.note(action, result)

    def close(self):
        try:
            self.collection.close()
        finally:
            super().close()

    def record_tool_sequence(self, task, color, steps):
        """Run a model-planned tool program without model waits between steps."""
        from agent.primitive_tools import RECORDABLE_TOOLS
        from agent.tool_arguments import has_reference, resolve_references
        from agent.tools import build_tools, result_from_exception, validate_arguments

        action, t0 = "record_tool_sequence", time.monotonic()
        allowed = set(RECORDABLE_TOOLS)
        if (not isinstance(task, str) or not task.strip() or len(task) > self.cfg.agent.collection.max_task_text_chars
                or not isinstance(color, str) or not color.strip()
                or not isinstance(steps, list)
                or not 1 <= len(steps) <= self.cfg.agent.collection.max_steps):
            return self._fail(action, "A task sentence, nonempty label and bounded steps are required",
                              "invalid_arguments")
        definitions = {tool.spec.name: tool for tool in build_tools(self.cfg)}
        planned = []
        for index, step in enumerate(steps):
            if not isinstance(step, dict) or set(step) != {"name", "arguments"}:
                return self._fail(action, f"Invalid step {index + 1}", "invalid_arguments")
            name, args = step["name"], step["arguments"]
            if not isinstance(name, str) or name not in allowed or name not in definitions:
                return self._fail(action, f"Tool {name!r} is not recordable", "invalid_arguments")
            error = None if has_reference(args) else validate_arguments(definitions[name].spec.input_schema, args)
            if error is not None:
                return self._fail(action, f"Step {index + 1}: {error}", "invalid_arguments")
            planned.append((name, dict(args), definitions[name]))

        try:
            self.collection.begin(color, task_text=task.strip(), sequence_mode=True)
        except ImportError as exc:
            return self._fail(action, f"Dataset dependencies missing: {exc}", "disabled")
        except ValueError as exc:
            return self._fail(action, str(exc))

        results = []
        references = {}
        try:
            for index, (name, args, definition) in enumerate(planned, 1):
                try:
                    resolved = resolve_references(args, references)
                    error = validate_arguments(definition.spec.input_schema, resolved)
                    if error:
                        result = self._fail(name, error, "invalid_arguments")
                    else:
                        result = definition.run(self, resolved)
                except Exception as exc:  # return the failed step, then stop the program
                    result = result_from_exception(name, exc)
                references[f"step{index}"] = result.to_envelope()
                self.collection.note(name, result)
                results.append({"step": index, "tool": name, **result.to_envelope()})
                if not result.ok or not self.collection.recording:
                    return self._result(False, action, result.reason if not result.ok else "task_incomplete",
                                        (f"Step {index} ({name}) stopped: {result.detail}" if not result.ok else
                                         f"Step {index} completed, but recording was discarded: {self.collection.last_error}"),
                                        t0=t0, step_results=results,
                                        collection=self.collection.status())
            try:
                saved = self.collection.save_sequence()
            except ValueError as exc:
                return self._result(False, action, "task_incomplete", str(exc), t0=t0,
                                    step_results=results, collection=self.collection.status())
            return self._result(saved, action, "ok" if saved else "task_incomplete",
                                "Recorded tool sequence saved." if saved else "Recorded take was discarded.",
                                t0=t0, step_results=results, collection=self.collection.status())
        finally:
            if self.collection.recorder and self.collection.recorder.is_open:
                self.collection.discard("sequence_interrupted")

    def begin_episode(self, object_id, observation_id, task=None):
        try:
            block = self._object(object_id, observation_id)
            self.collection.begin(block.color, task_text=task or (
                None if block.color in self.cfg.task3.task_templates else "Manipulate the selected block"))
        except ImportError as exc:
            return self._fail("begin_episode", f"Dataset dependencies missing: {exc}", "disabled")
        except ValueError as exc:
            return self._fail("begin_episode", str(exc))
        return self._result(True, "begin_episode", "ok", collection=self.collection.status())

    def save_episode(self):
        try:
            saved = self.collection.save()
        except ValueError as exc:
            return self._fail("save_episode", str(exc))
        return self._result(saved, "save_episode", "ok" if saved else "precondition", collection=self.collection.status())

    def discard_episode(self, reason):
        self.collection.discard(reason)
        return self._result(True, "discard_episode", "ok", collection=self.collection.status())

    def collection_status(self):
        return self._result(True, "collection_status", "ok", collection=self.collection.status())

    def finish_dataset(self):
        try:
            status = self.collection.finish()
        except ValueError as exc:
            return self._fail("finish_dataset", str(exc))
        return self._result(True, "finish_dataset", "ok", collection=status)
