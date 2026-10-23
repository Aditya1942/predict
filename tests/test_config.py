from pathlib import Path

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
