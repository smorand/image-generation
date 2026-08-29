"""Smoke tests for the `info` command on both CLIs."""

from typer.testing import CliRunner

runner = CliRunner()


def test_image_gen_info_runs_and_lists_schedulers():
    from image_gen.cli import app

    result = runner.invoke(app, ["info"])

    assert result.exit_code == 0, result.output
    assert "euler_a" in result.output
    assert "Default Negative Prompt" in result.output


def test_video_gen_info_runs_and_lists_backends():
    from video_gen.cli import app

    result = runner.invoke(app, ["info"])

    assert result.exit_code == 0, result.output
    assert "ltx" in result.output
    assert "wan" in result.output
    assert "Default negative prompt" in result.output
