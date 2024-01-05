"""Test Module for ConfigurationManager in fraudDetection.config.configuration

This module contains test functions for the ConfigurationManager class
in the fraudDetection.config.configuration module.

"""

from pathlib import Path
import pytest
from unittest.mock import Mock, patch
from fraudDetection.config.configuration import ConfigurationManager, CONFIG_FILE_PATH


@pytest.fixture
def mock_read_yaml(monkeypatch):
    """Fixture for mocking the read_yaml function."""
    with patch("fraudDetection.config.configuration.read_yaml") as mock_read_yaml:
        monkeypatch.setattr(ConfigurationManager, "read_yaml", mock_read_yaml)
        yield mock_read_yaml


def test_get_model_pusher_config(mock_read_yaml):
    """Test the get_model_pusher_config method of ConfigurationManager."""
    # Set up the mock behavior for read_yaml
    mock_read_yaml.return_value = {
        "MODEL_PUSHER_CONFIG_KEY": {"MODEL_PUSHER_EXPORT_DIR_KEY": "export_dir"}
    }

    # Set up the configuration manager
    config_manager = ConfigurationManager(CONFIG_FILE_PATH)

    # Define the expected configuration
    expected_export_dir_path = Path("mocked_export_dir/mock_timestamp")
    expected_saved_models_directory = Path("mocked_saved_models_directory")

    # Mock the values used in the method
    config_manager.artifact_dir = Path("mocked_artifact_dir")
    config_manager.current_time_stamp = "mock_timestamp"

    # Call the method to get the configuration
    model_pusher_config = config_manager.get_model_pusher_config()

    # Assert that the returned ModelPusherConfig matches the expected configuration
    assert model_pusher_config.model_export_dir == expected_export_dir_path
    assert model_pusher_config.saved_models_directory == expected_saved_models_directory


def test_get_model_pusher_config_exception(mock_read_yaml, monkeypatch):
    """Test the get_model_pusher_config method when an exception is raised."""
    # Set up the mock behavior for read_yaml to raise an exception
    mock_read_yaml.side_effect = Exception("Mocked exception")

    # Set up the configuration manager
    config_manager = ConfigurationManager(CONFIG_FILE_PATH)

    # Call the method and assert that it raises the expected exception
    with pytest.raises(Exception, match="Mocked exception"):
        config_manager.get_model_pusher_config()
