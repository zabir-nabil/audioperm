
### fork
 * fork the repository
### testing
 * run from audioperm root, `python -m pytest tests/`
 * add new tests for added feature.
 * make sure all unit tests pass.
### merge
 * send a merge request with a summary of changes
### versioning [CAUTION :volcano:]
 * bump the version number in `audioperm/__init__.py` in accordance with the semantic versioning specification
### wheel [CAUTION :bomb:]
 * `python3 setup.py sdist bdist_wheel`
 * `python3 -m twine upload dist/*`
 * commit and push the change with a commit message like this: "Release vx.y.z" (replace x.y.z with the package version)
 * add and push a git tag to the release commit

