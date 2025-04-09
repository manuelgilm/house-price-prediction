from package.utils.file_system import get_root_project_path
from package.data.feature_processing import process_textual_data
from package.data.feature_processing import process_images
from package.data.feature_processing import create_feature_metadata_table


def test():
    create_feature_metadata_table()
