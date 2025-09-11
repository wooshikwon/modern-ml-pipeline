from src.cli.utils.recipe_builder import RecipeBuilder


class UIStub:
    def __init__(self):
        self.non_empty_validator_invoked = False

    # No-op displays
    def show_panel(self, *args, **kwargs):
        pass
    def show_info(self, *args, **kwargs):
        pass
    def show_table(self, *args, **kwargs):
        pass
    def print_divider(self, *args, **kwargs):
        pass

    # Input utilities
    def select_from_list(self, title, options, show_numbers=True, allow_cancel=True):
        # Choose Timeseries task when selecting task
        if "Task" in title and "선택" in title and any(opt.lower() == "timeseries" for opt in options):
            return next(opt for opt in options if opt.lower() == "timeseries")
        # Otherwise pick first option deterministically
        return options[0]

    def confirm(self, message, default=False, show_default=True):
        # We avoid feature store path to focus on timeseries timestamp validator
        return False

    def number_input(self, prompt, default=None, min_value=None, max_value=None):
        return default if default is not None else (min_value or 0)

    def text_input(self, prompt, default=None, password=False, show_default=True, validator=None):
        # When the timeseries timestamp prompt appears, simulate empty → valid
        if "Timeseries 설정" in prompt:
            # This branch is not the actual text_input call, it's show_info call
            return default or ""

        if "Timestamp column 이름 (시계열" in prompt:
            if validator is not None:
                # First attempt: empty string should be rejected
                self.non_empty_validator_invoked = True
                ok = validator("")
                if ok:  # Should not happen
                    return ""
                # Second attempt: valid value
                ok2 = validator("timestamp")
                assert ok2 is True
            return "timestamp"

        # For other prompts, return default or a simple value
        if "Recipe 이름" in prompt:
            return "ts_recipe"
        if "SQL 파일 경로" in prompt or "데이터 파일 경로" in prompt:
            return "data/train.csv"
        if "Target column" in prompt:
            return "target"
        if "Entity column" in prompt:
            return "entity_id"

        return default or "test_value"


def test_recipe_builder_enforces_non_empty_timestamp_for_timeseries(monkeypatch):
    builder = RecipeBuilder()
    ui = UIStub()
    # Inject stub UI
    builder.ui = ui

    selections = builder.run_interactive_flow()

    # Validator path should have been exercised
    assert ui.non_empty_validator_invoked is True
    # And the final selection must contain the valid timestamp
    assert selections.get("timeseries_timestamp_column") == "timestamp"

