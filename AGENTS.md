# Project language

Use English for source code, identifiers, comments, docstrings, UI labels, CLI
messages and project documentation. Keep this convention when adding or editing
files, including historical reference code.

Communicate with the user in Croatian when they use Croatian. This does not
change the language used in project files.

# Runtime

Keep the application and its distribution native Rust/OpenCL. Do not introduce a
Python runtime dependency. Historical Python reference files are not part of the
native application or AppImage.

# Commits

Write commit messages as a single short line, with no body. Show the exact
message to the user before asking whether to commit.

Do not add `Co-Authored-By` or other AI attribution lines to commit messages
or pull request descriptions.

# README

Do not add to or change `README.md` unless the user explicitly asks for it.
