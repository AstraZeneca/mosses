.PHONY: build install upload upload_test test coverage

build:
	rm -rf build
	rm -rf dist
	python -m build --wheel
	python -m build --sdist

install:
	pip install dist/mosses-*.whl --force-reinstall

upload_test:
	# twine upload -r testpypi dist/*

upload:
	# twine upload -r pypi dist/*

coverage:
	pytest --cov=src test/ -vv --cov-report term-missing

create_git_tag:
	version=$$(grep -Po "version\s*=\s*['\"]\K[^'\"]+" pyproject.toml); \
	git tag v$$version
	git push --tags
