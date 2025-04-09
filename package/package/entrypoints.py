from package.utils.file_system import get_root_project_path


def test():
    root = get_root_project_path()
    print(root)
