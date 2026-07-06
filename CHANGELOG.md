# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI workflow for Python testing and linting
- Project preview SVG image
- GitHub badges (stars, forks, issues, license, last-commit)
- Tech stack badges (Python, PyTorch, Gymnasium, etc.)
- MIT LICENSE file

### Changed
- Updated README with model download instructions and cloud storage link template
- Added project preview screenshot section to README

## [1.1.0] - 2026-05-22

### Added
- Improved DQN agent training with convergence detection
- Debug tools for game flow and win detection
- Model evaluation functionality (2000+ episode evaluation)
- Enhanced logging system with detailed training metrics
- Auto-archiver for training checkpoints
- Training visualization with convergence curves
- Web UI for monitoring training progress

### Changed
- Optimized Mahjong environment for faster training
- Improved reward function with win priority
- Enhanced neural network architecture

### Fixed
- Win detection accuracy issues
- Game flow bugs in special hand patterns
- Action space representation

## [1.0.0] - 2026-05-01

### Added
- Initial DQN implementation for Mahjong AI
- Mahjong game environment with standard rules
- Basic training pipeline
- CLI interface for playing against the AI
- Model saving and loading functionality
- TensorBoard integration for training monitoring

[Unreleased]: https://github.com/lccuhk/rl-doraemon/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/lccuhk/rl-doraemon/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/lccuhk/rl-doraemon/releases/tag/v1.0.0
