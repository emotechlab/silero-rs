# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Initial version of inference and switching logic taken from internal Emotech
code
- Added in `end_silence_length` to track the raw end silences
- Added new pub function `validate_input` for `VadSession` struct. `process` function will use it to make sure input is valid in debug mode.
