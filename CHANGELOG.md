<!-- Entries in each category are sorted by merge time, with the latest PRs appearing first. -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on a mixture of [Keep a Changelog] and [Common Changelog].
This project adheres to [Semantic Versioning], with the exception that minor releases may include breaking changes.

## [Unreleased]

### Changed

- ⬆️ Update Catalyst to version 0.14.1 ([#77]) ([**@denialhaag**])

## [1.0.0] - 2026-01-26

### Added

- ⬆️ Add Python 3.14 support ([#45]) ([**@flowerthrower**], [**@burgholzer**])
- 🐍 Introduce Python package providing Catalyst plugin utilities and device configuration ([#20]) ([**@flowerthrower**])
- 🧪 Add comprehensive round-trip Python integration tests ([#20]) ([**@flowerthrower**])
- 🔌 Add MLIR plugin for connecting MQT Core with Catalyst ([#3]) ([**@flowerthrower**], [**@burgholzer**])
- 📦 Set up the initial repo structure and configuration ([#1]) ([**@flowerthrower**], [**@burgholzer**])

### Changed

- ⬆️ Update Catalyst to version 0.14.0 ([#45]) ([**@flowerthrower**], [**@burgholzer**])
- 🔄 Migrate testing infrastructure from LIT/MLIR-level to Python/pytest ([#20]) ([**@flowerthrower**])
- 👷 Update CI/CD macOS runners to `macos-15` ([#20]) ([**@flowerthrower**])
- 📦 Bump `mqt-core` to version 3.4.0 ([#20]) ([**@flowerthrower**])

### Removed

- 🗑️ Remove LIT/MLIR test infrastructure and files ([#20]) ([**@flowerthrower**])

## Initial discussions

_📚 Refer to the [original MQT Core PR] for initial discussions and decisions leading to this project._

<!-- Version links -->

[unreleased]: https://github.com/munich-quantum-toolkit/core-plugins-catalyst/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/munich-quantum-toolkit/core-plugins-catalyst/compare/052a747...v1.0.0

<!-- PR links -->

[#77]: https://github.com/munich-quantum-toolkit/core-plugins-catalyst/pull/77
[#45]: https://github.com/munich-quantum-toolkit/core-plugins-catalyst/pull/45
[#20]: https://github.com/munich-quantum-toolkit/core-plugins-catalyst/pull/20
[#3]: https://github.com/munich-quantum-toolkit/core-plugins-catalyst/pull/3
[#1]: https://github.com/munich-quantum-toolkit/core-plugins-catalyst/pull/1

<!-- Contributor -->

[**@flowerthrower**]: https://github.com/flowerthrower
[**@ystade**]: https://github.com/ystade
[**@burgholzer**]: https://github.com/burgholzer
[**@denialhaag**]: https://github.com/denialhaag

<!-- General links -->

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
[Common Changelog]: https://common-changelog.org
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
[original MQT Core PR]: https://github.com/munich-quantum-toolkit/core/pull/881
