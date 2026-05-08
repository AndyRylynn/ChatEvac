import json
import re


class EvacWorkflowManager:
    """
    Load state machine configuration from workflow_config.json.
    Each state maps to a single-letter symbol (A-O); the LLM triggers transitions by outputting symbols.
    """

    def __init__(self, config_path: str, verbose: bool = True):
        self._verbose = verbose
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"[EvacWorkflowManager] workflow_config.json not found at: {config_path}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"[EvacWorkflowManager] Invalid JSON in config: {e}")

        if "states" not in config:
            raise ValueError("[EvacWorkflowManager] Config must contain 'states'")

        self._states: dict = config["states"]
        self._re_execution_context: str = config.get("re_execution_context", "")
        self._global_re_execution_targets: set = set(config.get("global_re_execution_targets", []))
        self._null_symbol: str = config.get("null_action", {}).get("symbol", "Z")
        self.current_state: str = "idle"

        # Build symbol <-> state name bidirectional mapping (null_symbol Z not mapped to any state, handled separately)
        self._symbol_to_state: dict[str, str] = {}
        self._state_to_symbol: dict[str, str] = {}
        for state_name, state_cfg in self._states.items():
            symbol = state_cfg.get("symbol", "")
            if symbol:
                self._symbol_to_state[symbol] = state_name
                self._state_to_symbol[state_name] = symbol

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_status_text(self) -> str:
        return self._states[self.current_state]["description"]

    def get_symbol(self) -> str:
        """Return the single-letter symbol of the current state."""
        return self._state_to_symbol.get(self.current_state, "?")

    def get_state_by_symbol(self, symbol: str) -> str | None:
        """Return state name from symbol; returns None if not found."""
        return self._symbol_to_state.get(symbol.upper())

    def is_null_symbol(self, symbol: str) -> bool:
        """Check if symbol is the null action (does not trigger state transition)."""
        return symbol.upper() == self._null_symbol.upper()

    def get_all_states(self) -> dict:
        return {k: v["description"] for k, v in self._states.items()}

    # ------------------------------------------------------------------
    # Prompt injection (LLM semantic-driven)
    # ------------------------------------------------------------------

    def get_prompt_injection(self, user_text: str = "") -> str:
        """
        Generate a workflow context string to inject into the system_prompt.
        Includes current state, available transitions (symbol + trigger conditions), and re-execution hints.
        The LLM decides whether to output <WORKFLOW_ACTION>SYMBOL</WORKFLOW_ACTION> based on semantic understanding.
        """
        state_cfg = self._states[self.current_state]
        symbol = self.get_symbol()
        description = state_cfg["description"]
        hints = state_cfg.get("transition_hints", [])

        lines = [
            "",
            "[WORKFLOW STATE CONTEXT]",
            f"Current State: {symbol} — {description}",
        ]

        if hints:
            lines.append("Available Transitions:")
            for hint in hints:
                ts = hint["target_symbol"]
                td = self._states.get(hint["target_state"], {}).get("description", hint["target_state"])
                trigger = hint["trigger"]
                lines.append(f"  → {ts} ({td}): Output <WORKFLOW_ACTION>{ts}</WORKFLOW_ACTION>")
                lines.append(f"    Trigger when: {trigger}")

        if self._re_execution_context:
            lines.append(f"Re-execution: {self._re_execution_context}")

        lines.append(
            f"If the user's message is unrelated to the evacuation workflow (e.g., general questions, "
            f"small talk, or anything outside the assessment scope), output "
            f"<WORKFLOW_ACTION>{self._null_symbol}</WORKFLOW_ACTION> and respond freely. "
            f"The workflow state will remain unchanged."
        )
        lines.append(
            "INSTRUCTION: Evaluate the user's message semantically. "
            "If their intent matches a transition condition above, include the "
            f"corresponding <WORKFLOW_ACTION>SYMBOL</WORKFLOW_ACTION> tag. "
            f"If it does not match any transition, use <WORKFLOW_ACTION>{self._null_symbol}</WORKFLOW_ACTION>."
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def can_transition(self, target_state: str) -> bool:
        """
        Check whether transition from current state to target_state is allowed.
        Two cases are allowed:
          1. target_state is in the current state's allowed_transitions whitelist (sequential flow)
          2. target_state is in global_re_execution_targets (re-execution targets available from any state)
        """
        allowed = self._states[self.current_state].get("allowed_transitions", [])
        return target_state in allowed or target_state in self._global_re_execution_targets

    def transition_to(self, target_state: str) -> bool:
        if target_state not in self._states:
            if self._verbose:
                print(f"[WorkflowEngine] ERROR: Unknown state '{target_state}'")
            return False
        if not self.can_transition(target_state):
            if self._verbose:
                print(
                    f"[WorkflowEngine] WARN: {self.current_state} -> {target_state} "
                    "not in allowed_transitions"
                )
            return False
        old = self.current_state
        self.current_state = target_state
        if self._verbose:
            print(f"[WorkflowEngine] State: {old} -> {target_state}")
        return True

    def force_transition_to(self, target_state: str) -> None:
        """Force a transition bypassing the whitelist; used for system-triggered state changes and error recovery."""
        if target_state not in self._states:
            if self._verbose:
                print(f"[WorkflowEngine] ERROR: Unknown state '{target_state}'")
            return
        old = self.current_state
        self.current_state = target_state
        if self._verbose:
            print(f"[WorkflowEngine] Force state: {old} -> {target_state}")

    def transition_to_symbol(self, symbol: str) -> bool:
        """Trigger a state transition by symbol; returns whether it succeeded."""
        target_state = self.get_state_by_symbol(symbol)
        if not target_state:
            if self._verbose:
                print(f"[WorkflowEngine] ERROR: Unknown symbol '{symbol}'")
            return False
        return self.transition_to(target_state)
