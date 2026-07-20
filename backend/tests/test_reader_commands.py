import pytest

from multimodal_lab.run_evaluate_reader import build_parser as evaluation_parser
from multimodal_lab.run_train_reader import build_parser as training_parser


def test_training_command_has_no_test_split_argument():
    parser = training_parser()
    destinations = {action.dest for action in parser._actions}

    assert "split" not in destinations
    assert "checkpoint" not in destinations


def test_test_command_requires_an_expected_checkpoint_hash():
    parser = evaluation_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--config",
                "config.json",
                "--checkpoint",
                "reader.pt",
                "--output-dir",
                "runs",
            ]
        )
