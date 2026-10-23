from pathlib import Path

import pytest

from predict.config import DEFAULT_HOME, Settings


def settings(**values) -> Settings:
    return Settings(_env_file=None, football_data_org_token="token", **values)


def test_predict_home_defaults_to_a_folder_in_the_home_directory():
    assert settings().predict_home == DEFAULT_HOME == Path.home() / ".predict"


def test_predict_home_expands_a_tilde():
    assert settings(predict_home="~/elsewhere").predict_home == Path.home() / "elsewhere"


def test_predict_home_from_the_environment_expands_a_tilde(monkeypatch):
    monkeypatch.setenv("PREDICT_HOME", "~/from-env")
    assert settings().predict_home == Path.home() / "from-env"


def test_an_absolute_predict_home_is_kept(tmp_path):
    assert settings(predict_home=tmp_path).predict_home == tmp_path


def _env_file(folder, *lines: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ".env").write_text("\n".join(lines) + "\n")


@pytest.fixture
def folders(monkeypatch, tmp_path):
    """A fake home and a working directory, with the Settings variables cleared from the environment."""
    home, cwd = tmp_path / "home", tmp_path / "cwd"
    cwd.mkdir(parents=True)
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)
    for name in ("FOOTBALL_DATA_ORG_TOKEN", "ODDS_API_KEY", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return home / ".predict", cwd


def test_settings_are_read_from_the_dot_env_in_the_predict_home(folders):
    predict_home, _ = folders
    _env_file(predict_home, "FOOTBALL_DATA_ORG_TOKEN=from-home", "ODDS_API_KEY=odds")
    loaded = Settings()
    assert (loaded.football_data_org_token, loaded.odds_api_key) == ("from-home", "odds")


def test_settings_are_read_from_the_dot_env_in_the_working_directory(folders):
    _, cwd = folders
    _env_file(cwd, "FOOTBALL_DATA_ORG_TOKEN=from-cwd")
    assert Settings().football_data_org_token == "from-cwd"


def test_the_working_directory_dot_env_wins_and_the_home_one_fills_the_rest(folders):
    predict_home, cwd = folders
    _env_file(predict_home, "FOOTBALL_DATA_ORG_TOKEN=from-home", "TYPESAFE_API_KEY=typesafe")
    _env_file(cwd, "FOOTBALL_DATA_ORG_TOKEN=from-cwd")
    loaded = Settings()
    assert (loaded.football_data_org_token, loaded.typesafe_api_key) == ("from-cwd", "typesafe")
