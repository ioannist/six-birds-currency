import currencymorphism


def test_import_currencymorphism() -> None:
    assert currencymorphism is not None


def test_version_exists() -> None:
    assert hasattr(currencymorphism, "__version__")
