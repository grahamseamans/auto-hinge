# UV Documentation

# Overview and Quick Start

# uv

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![image](https://img.shields.io/pypi/v/uv.svg)](https://pypi.python.org/pypi/uv)
[![image](https://img.shields.io/pypi/l/uv.svg)](https://pypi.python.org/pypi/uv)
[![image](https://img.shields.io/pypi/pyversions/uv.svg)](https://pypi.python.org/pypi/uv)
[![Actions status](https://github.com/astral-sh/uv/actions/workflows/ci.yml/badge.svg)](https://github.com/astral-sh/uv/actions)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/astral-sh)

An extremely fast Python package and project manager, written in Rust.

<p align="center">
  <picture align="center">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/astral-sh/uv/assets/1309177/03aa9163-1c79-4a87-a31d-7a9311ed9310">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/astral-sh/uv/assets/1309177/629e59c0-9c6e-4013-9ad4-adb2bcf5080d">
    <img alt="Shows a bar chart with benchmark results." src="https://github.com/astral-sh/uv/assets/1309177/629e59c0-9c6e-4013-9ad4-adb2bcf5080d">
  </picture>
</p>

<p align="center">
  <i>Installing <a href="https://trio.readthedocs.io/">Trio</a>'s dependencies with a warm cache.</i>
</p>

## Highlights

- 🚀 A single tool to replace `pip`, `pip-tools`, `pipx`, `poetry`, `pyenv`, `twine`, `virtualenv`,
  and more.
- ⚡️ [10-100x faster](https://github.com/astral-sh/uv/blob/main/BENCHMARKS.md) than `pip`.
- 🗂️ Provides [comprehensive project management](#projects), with a
  [universal lockfile](https://docs.astral.sh/uv/concepts/projects/layout#the-lockfile).
- ❇️ [Runs scripts](#scripts), with support for
  [inline dependency metadata](https://docs.astral.sh/uv/guides/scripts#declaring-script-dependencies).
- 🐍 [Installs and manages](#python-versions) Python versions.
- 🛠️ [Runs and installs](#tools) tools published as Python packages.
- 🔩 Includes a [pip-compatible interface](#the-pip-interface) for a performance boost with a
  familiar CLI.
- 🏢 Supports Cargo-style [workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces) for
  scalable projects.
- 💾 Disk-space efficient, with a [global cache](https://docs.astral.sh/uv/concepts/cache) for
  dependency deduplication.
- ⏬ Installable without Rust or Python via `curl` or `pip`.
- 🖥️ Supports macOS, Linux, and Windows.

uv is backed by [Astral](https://astral.sh), the creators of
[Ruff](https://github.com/astral-sh/ruff).

## Installation

Install uv with our standalone installers:

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or, from [PyPI](https://pypi.org/project/uv/):

```bash
# With pip.
pip install uv
```

```bash
# Or pipx.
pipx install uv
```

If installed via the standalone installer, uv can update itself to the latest version:

```bash
uv self update
```

See the [installation documentation](https://docs.astral.sh/uv/getting-started/installation/) for
details and alternative installation methods.

## Documentation

uv's documentation is available at [docs.astral.sh/uv](https://docs.astral.sh/uv).

Additionally, the command line reference documentation can be viewed with `uv help`.

## Features

### Projects

uv manages project dependencies and environments, with support for lockfiles, workspaces, and more,
similar to `rye` or `poetry`:

```console
$ uv init example
Initialized project `example` at `/home/user/example`

$ cd example

$ uv add ruff
Creating virtual environment at: .venv
Resolved 2 packages in 170ms
   Built example @ file:///home/user/example
Prepared 2 packages in 627ms
Installed 2 packages in 1ms
 + example==0.1.0 (from file:///home/user/example)
 + ruff==0.5.0

$ uv run ruff check
All checks passed!

$ uv lock
Resolved 2 packages in 0.33ms

$ uv sync
Resolved 2 packages in 0.70ms
Audited 1 package in 0.02ms
```

See the [project documentation](https://docs.astral.sh/uv/guides/projects/) to get started.

uv also supports building and publishing projects, even if they're not managed with uv. See the
[publish guide](https://docs.astral.sh/uv/guides/publish/) to learn more.

### Scripts

uv manages dependencies and environments for single-file scripts.

Create a new script and add inline metadata declaring its dependencies:

```console
$ echo 'import requests; print(requests.get("https://astral.sh"))' > example.py

$ uv add --script example.py requests
Updated `example.py`
```

Then, run the script in an isolated virtual environment:

```console
$ uv run example.py
Reading inline script metadata from: example.py
Installed 5 packages in 12ms
<Response [200]>
```

See the [scripts documentation](https://docs.astral.sh/uv/guides/scripts/) to get started.

### Tools

uv executes and installs command-line tools provided by Python packages, similar to `pipx`.

Run a tool in an ephemeral environment using `uvx` (an alias for `uv tool run`):

```console
$ uvx pycowsay 'hello world!'
Resolved 1 package in 167ms
Installed 1 package in 9ms
 + pycowsay==0.0.0.2
  """

  ------------
< hello world! >
  ------------
   \   ^__^
    \  (oo)\_______
       (__)\       )\/\
           ||----w |
           ||     ||
```

Install a tool with `uv tool install`:

```console
$ uv tool install ruff
Resolved 1 package in 6ms
Installed 1 package in 2ms
 + ruff==0.5.0
Installed 1 executable: ruff

$ ruff --version
ruff 0.5.0
```

See the [tools documentation](https://docs.astral.sh/uv/guides/tools/) to get started.

### Python versions

uv installs Python and allows quickly switching between versions.

Install multiple Python versions:

```console
$ uv python install 3.10 3.11 3.12
Searching for Python versions matching: Python 3.10
Searching for Python versions matching: Python 3.11
Searching for Python versions matching: Python 3.12
Installed 3 versions in 3.42s
 + cpython-3.10.14-macos-aarch64-none
 + cpython-3.11.9-macos-aarch64-none
 + cpython-3.12.4-macos-aarch64-none
```

Download Python versions as needed:

```console
$ uv venv --python 3.12.0
Using Python 3.12.0
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate

$ uv run --python pypy@3.8 -- python --version
Python 3.8.16 (a9dbdca6fc3286b0addd2240f11d97d8e8de187a, Dec 29 2022, 11:45:30)
[PyPy 7.3.11 with GCC Apple LLVM 13.1.6 (clang-1316.0.21.2.5)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>>
```

Use a specific Python version in the current directory:

```console
$ uv python pin 3.11
Pinned `.python-version` to `3.11`
```

See the [Python installation documentation](https://docs.astral.sh/uv/guides/install-python/) to get
started.

### The pip interface

uv provides a drop-in replacement for common `pip`, `pip-tools`, and `virtualenv` commands.

uv extends their interfaces with advanced features, such as dependency version overrides,
platform-independent resolutions, reproducible resolutions, alternative resolution strategies, and
more.

Migrate to uv without changing your existing workflows — and experience a 10-100x speedup — with the
`uv pip` interface.

Compile requirements into a platform-independent requirements file:

```console
$ uv pip compile docs/requirements.in \
   --universal \
   --output-file docs/requirements.txt
Resolved 43 packages in 12ms
```

Create a virtual environment:

```console
$ uv venv
Using Python 3.12.3
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate
```

Install the locked requirements:

```console
$ uv pip sync docs/requirements.txt
Resolved 43 packages in 11ms
Installed 43 packages in 208ms
 + babel==2.15.0
 + black==24.4.2
 + certifi==2024.7.4
 ...
```

See the [pip interface documentation](https://docs.astral.sh/uv/pip/index/) to get started.

## Platform support

See uv's [platform support](https://docs.astral.sh/uv/reference/platforms/) document.

## Versioning policy

See uv's [versioning policy](https://docs.astral.sh/uv/reference/versioning/) document.

## Contributing

We are passionate about supporting contributors of all levels of experience and would love to see
you get involved in the project. See the
[contributing guide](https://github.com/astral-sh/uv/blob/main/CONTRIBUTING.md) to get started.

## Acknowledgements

uv's dependency resolver uses [PubGrub](https://github.com/pubgrub-rs/pubgrub) under the hood. We're
grateful to the PubGrub maintainers, especially [Jacob Finkelman](https://github.com/Eh2406), for
their support.

uv's Git implementation is based on [Cargo](https://github.com/rust-lang/cargo).

Some of uv's optimizations are inspired by the great work we've seen in [pnpm](https://pnpm.io/),
[Orogene](https://github.com/orogene/orogene), and [Bun](https://github.com/oven-sh/bun). We've also
learned a lot from Nathaniel J. Smith's [Posy](https://github.com/njsmith/posy) and adapted its
[trampoline](https://github.com/njsmith/posy/tree/main/src/trampolines/windows-trampolines/posy-trampoline)
for Windows support.

## License

uv is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in uv
by you, as defined in the Apache-2.0 license, shall be dually licensed as above, without any
additional terms or conditions.

<div align="center">
  <a target="_blank" href="https://astral.sh" style="background:none">
    <img src="https://raw.githubusercontent.com/astral-sh/uv/main/assets/svg/Astral.svg" alt="Made by Astral">
  </a>
</div>

# Getting Started Guide

# Getting started

To help you get started with uv, we'll cover a few important topics:

- [Installing uv](./installation.md)
- [First steps after installation](./first-steps.md)
- [An overview of uv's features](./features.md)
- [How to get help](./help.md)

Read on, or jump ahead to another section:

- Get going quickly with [guides](../guides/index.md) for common workflows.
- Learn more about the core [concepts](../concepts/index.md) in uv.
- Use the [reference](../reference/index.md) documentation to find details about something specific.

# First Steps with UV

# First steps with uv

After [installing uv](./installation.md), you can check that uv is available by running the `uv`
command:

```console
$ uv
An extremely fast Python package manager.

Usage: uv [OPTIONS] <COMMAND>

...
```

You should see a help menu listing the available commands.

## Next steps

Now that you've confirmed uv is installed, check out an [overview of features](./features.md), learn
how to [get help](./help.md) if you run into any problems, or jump to the
[guides](../guides/index.md) to start using uv.

# Core Concepts Overview

# Concepts overview

Read the concept documents to learn more about uv's features:

- [Projects](./projects/index.md)
- [Tools](./tools.md)
- [Python versions](./python-versions.md)
- [Resolution](./resolution.md)
- [Caching](./cache.md)

Looking for a quick introduction to features? See the [guides](../guides/index.md) instead.

# Python Version Management

# Python versions

A Python version is composed of a Python interpreter (i.e. the `python` executable), the standard
library, and other supporting files.

## Managed and system Python installations

Since it is common for a system to have an existing Python installation, uv supports
[discovering](#discovery-of-python-versions) Python versions. However, uv also supports
[installing Python versions](#installing-a-python-version) itself. To distinguish between these two
types of Python installations, uv refers to Python versions it installs as _managed_ Python
installations and all other Python installations as _system_ Python installations.

!!! note

    uv does not distinguish between Python versions installed by the operating system vs those
    installed and managed by other tools. For example, if a Python installation is managed with
    `pyenv`, it would still be considered a _system_ Python version in uv.

## Requesting a version

A specific Python version can be requested with the `--python` flag in most uv commands. For
example, when creating a virtual environment:

```console
$ uv venv --python 3.11.6
```

uv will ensure that Python 3.11.6 is available — downloading and installing it if necessary — then
create the virtual environment with it.

The following Python version request formats are supported:

- `<version>` (e.g., `3`, `3.12`, `3.12.3`)
- `<version-specifier>` (e.g., `>=3.12,<3.13`)
- `<implementation>` (e.g., `cpython` or `cp`)
- `<implementation>@<version>` (e.g., `cpython@3.12`)
- `<implementation><version>` (e.g., `cpython3.12` or `cp312`)
- `<implementation><version-specifier>` (e.g., `cpython>=3.12,<3.13`)
- `<implementation>-<version>-<os>-<arch>-<libc>` (e.g., `cpython-3.12.3-macos-aarch64-none`)

Additionally, a specific system Python interpreter can be requested with:

- `<executable-path>` (e.g., `/opt/homebrew/bin/python3`)
- `<executable-name>` (e.g., `mypython3`)
- `<install-dir>` (e.g., `/some/environment/`)

By default, uv will automatically download Python versions if they cannot be found on the system.
This behavior can be
[disabled with the `python-downloads` option](#disabling-automatic-python-downloads).

### Python version files

The `.python-version` file can be used to create a default Python version request. uv searches for a
`.python-version` file in the working directory and each of its parents. If none is found, uv will
check the user-level configuration directory. Any of the request formats described above can be
used, though use of a version number is recommended for interoperability with other tools.

A `.python-version` file can be created in the current directory with the
[`uv python pin`](../reference/cli.md/#uv-python-pin) command.

A global `.python-version` file can be created in the user configuration directory with the
[`uv python pin --global`](../reference/cli.md/#uv-python-pin) command.

Discovery of `.python-version` files can be disabled with `--no-config`.

uv will not search for `.python-version` files beyond project or workspace boundaries (with the
exception of the user configuration directory).

## Installing a Python version

uv bundles a list of downloadable CPython and PyPy distributions for macOS, Linux, and Windows.

!!! tip

    By default, Python versions are automatically downloaded as needed without using
    `uv python install`.

To install a Python version at a specific version:

```console
$ uv python install 3.12.3
```

To install the latest patch version:

```console
$ uv python install 3.12
```

To install a version that satisfies constraints:

```console
$ uv python install '>=3.8,<3.10'
```

To install multiple versions:

```console
$ uv python install 3.9 3.10 3.11
```

To install a specific implementation:

```console
$ uv python install pypy
```

All of the [Python version request](#requesting-a-version) formats are supported except those that
are used for requesting local interpreters such as a file path.

By default `uv python install` will verify that a managed Python version is installed or install the
latest version. If a `.python-version` file is present, uv will install the Python version listed in
the file. A project that requires multiple Python versions may define a `.python-versions` file. If
present, uv will install all of the Python versions listed in the file.

!!! important

    The available Python versions are frozen for each uv release. To install new Python versions,
    you may need upgrade uv.

### Installing Python executables

!!! important

    Support for installing Python executables is in _preview_, this means the behavior is experimental
    and subject to change.

To install Python executables into your `PATH`, provide the `--preview` option:

```console
$ uv python install 3.12 --preview
```

This will install a Python executable for the requested version into `~/.local/bin`, e.g., as
`python3.12`.

!!! tip

    If `~/.local/bin` is not in your `PATH`, you can add it with `uv tool update-shell`.

To install `python` and `python3` executables, include the `--default` option:

```console
$ uv python install 3.12 --default --preview
```

When installing Python executables, uv will only overwrite an existing executable if it is managed
by uv — e.g., if `~/.local/bin/python3.12` exists already uv will not overwrite it without the
`--force` flag.

uv will update executables that it manages. However, it will prefer the latest patch version of each
Python minor version by default. For example:

```console
$ uv python install 3.12.7 --preview  # Adds `python3.12` to `~/.local/bin`
$ uv python install 3.12.6 --preview  # Does not update `python3.12`
$ uv python install 3.12.8 --preview  # Updates `python3.12` to point to 3.12.8
```

## Project Python versions

uv will respect Python requirements defined in `requires-python` in the `pyproject.toml` file during
project command invocations. The first Python version that is compatible with the requirement will
be used, unless a version is otherwise requested, e.g., via a `.python-version` file or the
`--python` flag.

## Viewing available Python versions

To list installed and available Python versions:

```console
$ uv python list
```

To filter the Python versions, provide a request, e.g., to show all Python 3.13 interpreters:

```console
$ uv python list 3.13
```

Or, to show all PyPy interpreters:

```console
$ uv python list pypy
```

By default, downloads for other platforms and old patch versions are hidden.

To view all versions:

```console
$ uv python list --all-versions
```

To view Python versions for other platforms:

```console
$ uv python list --all-platforms
```

To exclude downloads and only show installed Python versions:

```console
$ uv python list --only-installed
```

See the [`uv python list`](../reference/cli.md#uv-python-list) reference for more details.

## Finding a Python executable

To find a Python executable, use the `uv python find` command:

```console
$ uv python find
```

By default, this will display the path to the first available Python executable. See the
[discovery rules](#discovery-of-python-versions) for details about how executables are discovered.

This interface also supports many [request formats](#requesting-a-version), e.g., to find a Python
executable that has a version of 3.11 or newer:

```console
$ uv python find '>=3.11'
```

By default, `uv python find` will include Python versions from virtual environments. If a `.venv`
directory is found in the working directory or any of the parent directories or the `VIRTUAL_ENV`
environment variable is set, it will take precedence over any Python executables on the `PATH`.

To ignore virtual environments, use the `--system` flag:

```console
$ uv python find --system
```

## Discovery of Python versions

When searching for a Python version, the following locations are checked:

- Managed Python installations in the `UV_PYTHON_INSTALL_DIR`.
- A Python interpreter on the `PATH` as `python`, `python3`, or `python3.x` on macOS and Linux, or
  `python.exe` on Windows.
- On Windows, the Python interpreters in the Windows registry and Microsoft Store Python
  interpreters (see `py --list-paths`) that match the requested version.

In some cases, uv allows using a Python version from a virtual environment. In this case, the
virtual environment's interpreter will be checked for compatibility with the request before
searching for an installation as described above. See the
[pip-compatible virtual environment discovery](../pip/environments.md#discovery-of-python-environments)
documentation for details.

When performing discovery, non-executable files will be ignored. Each discovered executable is
queried for metadata to ensure it meets the [requested Python version](#requesting-a-version). If
the query fails, the executable will be skipped. If the executable satisfies the request, it is used
without inspecting additional executables.

When searching for a managed Python version, uv will prefer newer versions first. When searching for
a system Python version, uv will use the first compatible version — not the newest version.

If a Python version cannot be found on the system, uv will check for a compatible managed Python
version download.

### Python pre-releases

Python pre-releases will not be selected by default. Python pre-releases will be used if there is no
other available installation matching the request. For example, if only a pre-release version is
available it will be used but otherwise a stable release version will be used. Similarly, if the
path to a pre-release Python executable is provided then no other Python version matches the request
and the pre-release version will be used.

If a pre-release Python version is available and matches the request, uv will not download a stable
Python version instead.

## Disabling automatic Python downloads

By default, uv will automatically download Python versions when needed.

The [`python-downloads`](../reference/settings.md#python-downloads) option can be used to disable
this behavior. By default, it is set to `automatic`; set to `manual` to only allow Python downloads
during `uv python install`.

!!! tip

    The `python-downloads` setting can be set in a
    [persistent configuration file](../configuration/files.md) to change the default behavior, or
    the `--no-python-downloads` flag can be passed to any uv command.

## Requiring or disabling managed Python versions

By default, uv will attempt to use Python versions found on the system and only download managed
Python versions when necessary. To ignore system Python versions, and only use managed Python
versions, use the `--managed-python` flag:

```console
$ uv python list --managed-python
```

Similarly, to ignore managed Python versions and only use system Python versions, use the
`--no-managed-python` flag:

```console
$ uv python list --no-managed-python
```

To change uv's default behavior in a configuration file, use the
[`python-preference` setting](#adjusting-python-version-preferences).

## Adjusting Python version preferences

The [`python-preference`](../reference/settings.md#python-preference) setting determines whether to
prefer using Python installations that are already present on the system, or those that are
downloaded and installed by uv.

By default, the `python-preference` is set to `managed` which prefers managed Python installations
over system Python installations. However, system Python installations are still preferred over
downloading a managed Python version.

The following alternative options are available:

- `only-managed`: Only use managed Python installations; never use system Python installations.
  Equivalent to `--managed-python`.
- `system`: Prefer system Python installations over managed Python installations.
- `only-system`: Only use system Python installations; never use managed Python installations.
  Equivalent to `--no-managed-python`.

!!! note

    Automatic Python version downloads can be [disabled](#disabling-automatic-python-downloads)
    without changing the preference.

## Python implementation support

uv supports the CPython, PyPy, and GraalPy Python implementations. If a Python implementation is not
supported, uv will fail to discover its interpreter.

The implementations may be requested with either the long or short name:

- CPython: `cpython`, `cp`
- PyPy: `pypy`, `pp`
- GraalPy: `graalpy`, `gp`

Implementation name requests are not case sensitive.

See the [Python version request](#requesting-a-version) documentation for more details on the
supported formats.

## Managed Python distributions

uv supports downloading and installing CPython and PyPy distributions.

### CPython distributions

As Python does not publish official distributable CPython binaries, uv instead uses pre-built
distributions from the Astral
[`python-build-standalone`](https://github.com/astral-sh/python-build-standalone) project.
`python-build-standalone` is also is used in many other Python projects, like
[Rye](https://github.com/astral-sh/rye), [Mise](https://mise.jdx.dev/lang/python.html), and
[bazelbuild/rules_python](https://github.com/bazelbuild/rules_python).

The uv Python distributions are self-contained, highly-portable, and performant. While Python can be
built from source, as in tools like `pyenv`, doing so requires preinstalled system dependencies, and
creating optimized, performant builds (e.g., with PGO and LTO enabled) is very slow.

These distributions have some behavior quirks, generally as a consequence of portability; see the
[`python-build-standalone` quirks](https://gregoryszorc.com/docs/python-build-standalone/main/quirks.html)
documentation for details. Additionally, some platforms may not be supported (e.g., distributions
are not yet available for musl Linux on ARM).

### PyPy distributions

PyPy distributions are provided by the PyPy project.

# Tool Management

# Tools

Tools are Python packages that provide command-line interfaces.

!!! note

    See the [tools guide](../guides/tools.md) for an introduction to working with the tools
    interface — this document discusses details of tool management.

## The `uv tool` interface

uv includes a dedicated interface for interacting with tools. Tools can be invoked without
installation using `uv tool run`, in which case their dependencies are installed in a temporary
virtual environment isolated from the current project.

Because it is very common to run tools without installing them, a `uvx` alias is provided for
`uv tool run` — the two commands are exactly equivalent. For brevity, the documentation will mostly
refer to `uvx` instead of `uv tool run`.

Tools can also be installed with `uv tool install`, in which case their executables are
[available on the `PATH`](#the-path) — an isolated virtual environment is still used, but it is not
removed when the command completes.

## Execution vs installation

In most cases, executing a tool with `uvx` is more appropriate than installing the tool. Installing
the tool is useful if you need the tool to be available to other programs on your system, e.g., if
some script you do not control requires the tool, or if you are in a Docker image and want to make
the tool available to users.

## Tool environments

When running a tool with `uvx`, a virtual environment is stored in the uv cache directory and is
treated as disposable, i.e., if you run `uv cache clean` the environment will be deleted. The
environment is only cached to reduce the overhead of repeated invocations. If the environment is
removed, a new one will be created automatically.

When installing a tool with `uv tool install`, a virtual environment is created in the uv tools
directory. The environment will not be removed unless the tool is uninstalled. If the environment is
manually deleted, the tool will fail to run.

## Tool versions

Unless a specific version is requested, `uv tool install` will install the latest available of the
requested tool. `uvx` will use the latest available version of the requested tool _on the first
invocation_. After that, `uvx` will use the cached version of the tool unless a different version is
requested, the cache is pruned, or the cache is refreshed.

For example, to run a specific version of Ruff:

```console
$ uvx ruff@0.6.0 --version
ruff 0.6.0
```

A subsequent invocation of `uvx` will use the latest, not the cached, version.

```console
$ uvx ruff --version
ruff 0.6.2
```

But, if a new version of Ruff was released, it would not be used unless the cache was refreshed.

To request the latest version of Ruff and refresh the cache, use the `@latest` suffix:

```console
$ uvx ruff@latest --version
0.6.2
```

Once a tool is installed with `uv tool install`, `uvx` will use the installed version by default.

For example, after installing an older version of Ruff:

```console
$ uv tool install ruff==0.5.0
```

The version of `ruff` and `uvx ruff` is the same:

```console
$ ruff --version
ruff 0.5.0
$ uvx ruff --version
ruff 0.5.0
```

However, you can ignore the installed version by requesting the latest version explicitly, e.g.:

```console
$ uvx ruff@latest --version
0.6.2
```

Or, by using the `--isolated` flag, which will avoid refreshing the cache but ignore the installed
version:

```console
$ uvx --isolated ruff --version
0.6.2
```

`uv tool install` will also respect the `{package}@{version}` and `{package}@latest` specifiers, as
in:

```console
$ uv tool install ruff@latest
$ uv tool install ruff@0.6.0
```

## Tools directory

By default, the uv tools directory is named `tools` and is in the uv application state directory,
e.g., `~/.local/share/uv/tools`. The location may be customized with the `UV_TOOL_DIR` environment
variable.

To display the path to the tool installation directory:

```console
$ uv tool dir
```

Tool environments are placed in a directory with the same name as the tool package, e.g.,
`.../tools/<name>`.

!!! important

    Tool environments are _not_ intended to be mutated directly. It is strongly recommended never to
    mutate a tool environment manually, e.g., with a `pip` operation.

## Upgrading tools

Tool environments may be upgraded via `uv tool upgrade`, or re-created entirely via subsequent
`uv tool install` operations.

To upgrade all packages in a tool environment

```console
$ uv tool upgrade black
```

To upgrade a single package in a tool environment:

```console
$ uv tool upgrade black --upgrade-package click
```

Tool upgrades will respect the version constraints provided when installing the tool. For example,
`uv tool install black >=23,<24` followed by `uv tool upgrade black` will upgrade Black to the
latest version in the range `>=23,<24`.

To instead replace the version constraints, reinstall the tool with `uv tool install`:

```console
$ uv tool install black>=24
```

Similarly, tool upgrades will retain the settings provided when installing the tool. For example,
`uv tool install black --prerelease allow` followed by `uv tool upgrade black` will retain the
`--prerelease allow` setting.

!!! note

    Tool upgrades will reinstall the tool executables, even if they have not changed.

To reinstall packages during upgrade, use the `--reinstall` and `--reinstall-package` options.

To reinstall all packages in a tool environment

```console
$ uv tool upgrade black --reinstall
```

To reinstall a single package in a tool environment:

```console
$ uv tool upgrade black --reinstall-package click
```

## Including additional dependencies

Additional packages can be included during tool execution:

```console
$ uvx --with <extra-package> <tool>
```

And, during tool installation:

```console
$ uv tool install --with <extra-package> <tool-package>
```

The `--with` option can be provided multiple times to include additional packages.

The `--with` option supports package specifications, so a specific version can be requested:

```console
$ uvx --with <extra-package>==<version> <tool-package>
```

If the requested version conflicts with the requirements of the tool package, package resolution
will fail and the command will error.

## Tool executables

Tool executables include all console entry points, script entry points, and binary scripts provided
by a Python package. Tool executables are symlinked into the `bin` directory on Unix and copied on
Windows.

### The `bin` directory

Executables are installed into the user `bin` directory following the XDG standard, e.g.,
`~/.local/bin`. Unlike other directory schemes in uv, the XDG standard is used on _all platforms_
notably including Windows and macOS — there is no clear alternative location to place executables on
these platforms. The installation directory is determined from the first available environment
variable:

- `$UV_TOOL_BIN_DIR`
- `$XDG_BIN_HOME`
- `$XDG_DATA_HOME/../bin`
- `$HOME/.local/bin`

Executables provided by dependencies of tool packages are not installed.

### The `PATH`

The `bin` directory must be in the `PATH` variable for tool executables to be available from the
shell. If it is not in the `PATH`, a warning will be displayed. The `uv tool update-shell` command
can be used to add the `bin` directory to the `PATH` in common shell configuration files.

### Overwriting executables

Installation of tools will not overwrite executables in the `bin` directory that were not previously
installed by uv. For example, if `pipx` has been used to install a tool, `uv tool install` will
fail. The `--force` flag can be used to override this behavior.

## Relationship to `uv run`

The invocation `uv tool run <name>` (or `uvx <name>`) is nearly equivalent to:

```console
$ uv run --no-project --with <name> -- <name>
```

However, there are a couple notable differences when using uv's tool interface:

- The `--with` option is not needed — the required package is inferred from the command name.
- The temporary environment is cached in a dedicated location.
- The `--no-project` flag is not needed — tools are always run isolated from the project.
- If a tool is already installed, `uv tool run` will use the installed version but `uv run` will
  not.

If the tool should not be isolated from the project, e.g., when running `pytest` or `mypy`, then
`uv run` should be used instead of `uv tool run`.

# Package Resolution

# Resolution

Resolution is the process of taking a list of requirements and converting them to a list of package
versions that fulfill the requirements. Resolution requires recursively searching for compatible
versions of packages, ensuring that the requested requirements are fulfilled and that the
requirements of the requested packages are compatible.

## Dependencies

Most projects and packages have dependencies. Dependencies are other packages that are necessary in
order for the current package to work. A package defines its dependencies as _requirements_, roughly
a combination of a package name and acceptable versions. The dependencies defined by the current
project are called _direct dependencies_. The dependencies added by each dependency of the current
project are called _indirect_ or _transitive dependencies_.

!!! note

    See the [dependency specifiers
    page](https://packaging.python.org/en/latest/specifications/dependency-specifiers/)
    in the Python Packaging documentation for details about dependencies.

## Basic examples

To help demonstrate the resolution process, consider the following dependencies:

<!-- prettier-ignore -->
- The project depends on `foo` and `bar`.
- `foo` has one version, 1.0.0:
    - `foo 1.0.0` depends on `lib>=1.0.0`.
- `bar` has one version, 1.0.0:
    - `bar 1.0.0` depends on `lib>=2.0.0`.
- `lib` has two versions, 1.0.0 and 2.0.0. Both versions have no dependencies.

In this example, the resolver must find a set of package versions which satisfies the project
requirements. Since there is only one version of both `foo` and `bar`, those will be used. The
resolution must also include the transitive dependencies, so a version of `lib` must be chosen.
`foo 1.0.0` allows all available versions of `lib`, but `bar 1.0.0` requires `lib>=2.0.0` so
`lib 2.0.0` must be used.

In some resolutions, there may be more than one valid solution. Consider the following dependencies:

<!-- prettier-ignore -->
- The project depends on `foo` and `bar`.
- `foo` has two versions, 1.0.0 and 2.0.0:
    - `foo 1.0.0` has no dependencies.
    - `foo 2.0.0` depends on `lib==2.0.0`.
- `bar` has two versions, 1.0.0 and 2.0.0:
    - `bar 1.0.0` has no dependencies.
    - `bar 2.0.0` depends on `lib==1.0.0`
- `lib` has two versions, 1.0.0 and 2.0.0. Both versions have no dependencies.

In this example, some version of both `foo` and `bar` must be selected; however, determining which
version requires considering the dependencies of each version of `foo` and `bar`. `foo 2.0.0` and
`bar 2.0.0` cannot be installed together as they conflict on their required version of `lib`, so the
resolver must select either `foo 1.0.0` (along with `bar 2.0.0`) or `bar 1.0.0` (along with
`foo 1.0.0`). Both are valid solutions, and different resolution algorithms may yield either result.

## Platform markers

Markers allow attaching an expression to requirements that indicate when the dependency should be
used. For example `bar ; python_version < "3.9"` indicates that `bar` should only be installed on
Python 3.8 and earlier.

Markers are used to adjust a package's dependencies based on the current environment or platform.
For example, markers can be used to modify dependencies by operating system, CPU architecture,
Python version, Python implementation, and more.

!!! note

    See the [environment
    markers](https://packaging.python.org/en/latest/specifications/dependency-specifiers/#environment-markers)
    section in the Python Packaging documentation for more details about markers.

Markers are important for resolution because their values change the required dependencies.
Typically, Python package resolvers use the markers of the _current_ platform to determine which
dependencies to use since the package is often being _installed_ on the current platform. However,
for _locking_ dependencies this is problematic — the lockfile would only work for developers using
the same platform the lockfile was created on. To solve this problem, platform-independent, or
"universal" resolvers exist.

uv supports both [platform-specific](#platform-specific-resolution) and
[universal](#universal-resolution) resolution.

## Platform-specific resolution

By default, uv's pip interface, i.e., [`uv pip compile`](../pip/compile.md), produces a resolution
that is platform-specific, like `pip-tools`. There is no way to use platform-specific resolution in
the uv's project interface.

uv also supports resolving for specific, alternate platforms and Python versions with the
`--python-platform` and `--python-version` options. For example, if using Python 3.12 on macOS,
`uv pip compile --python-platform linux --python-version 3.10 requirements.in` can be used to
produce a resolution for Python 3.10 on Linux instead. Unlike universal resolution, during
platform-specific resolution, the provided `--python-version` is the exact python version to use,
not a lower bound.

!!! note

    Python's environment markers expose far more information about the current machine
    than can be expressed by a simple `--python-platform` argument. For example, the `platform_version` marker
    on macOS includes the time at which the kernel was built, which can (in theory) be encoded in
    package requirements. uv's resolver makes a best-effort attempt to generate a resolution that is
    compatible with any machine running on the target `--python-platform`, which should be sufficient for
    most use cases, but may lose fidelity for complex package and platform combinations.

## Universal resolution

uv's lockfile (`uv.lock`) is created with a universal resolution and is portable across platforms.
This ensures that dependencies are locked for everyone working on the project, regardless of
operating system, architecture, and Python version. The uv lockfile is created and modified by
[project](../concepts/projects/index.md) commands such as `uv lock`, `uv sync`, and `uv add`.

Universal resolution is also available in uv's pip interface, i.e.,
[`uv pip compile`](../pip/compile.md), with the `--universal` flag. The resulting requirements file
will contain markers to indicate which platform each dependency is relevant for.

During universal resolution, a package may be listed multiple times with different versions or URLs
if different versions are needed for different platforms — the markers determine which version will
be used. A universal resolution is often more constrained than a platform-specific resolution, since
we need to take the requirements for all markers into account.

During universal resolution, all required packages must be compatible with the _entire_ range of
`requires-python` declared in the `pyproject.toml`. For example, if a project's `requires-python` is
`>=3.8`, resolution will fail if all versions of given dependency require Python 3.9 or later, since
the dependency lacks a usable version for (e.g.) Python 3.8, the lower bound of the project's
supported range. In other words, the project's `requires-python` must be a subset of the
`requires-python` of all its dependencies.

When selecting the compatible version for a given dependency, uv will
([by default](#multi-version-resolution)) attempt to choose the latest compatible version for each
supported Python version. For example, if a project's `requires-python` is `>=3.8`, and the latest
version of a dependency requires Python 3.9 or later, while all prior versions supporting Python
3.8, the resolver will select the latest version for users running Python 3.9 or later, and previous
versions for users running Python 3.8.

When evaluating `requires-python` ranges for dependencies, uv only considers lower bounds and
ignores upper bounds entirely. For example, `>=3.8, <4` is treated as `>=3.8`. Respecting upper
bounds on `requires-python` often leads to formally correct but practically incorrect resolutions,
as, e.g., resolvers will backtrack to the first published version that omits the upper bound (see:
[`Requires-Python` upper limits](https://discuss.python.org/t/requires-python-upper-limits/12663)).

### Limited resolution environments

By default, the universal resolver attempts to solve for all platforms and Python versions.

If your project supports only a limited set of platforms or Python versions, you can constrain the
set of solved platforms via the `environments` setting, which accepts a list of
[PEP 508 environment markers](https://packaging.python.org/en/latest/specifications/dependency-specifiers/#environment-markers).
In other words, you can use the `environments` setting to _reduce_ the set of supported platforms.

For example, to constrain the lockfile to macOS and Linux, and avoid solving for Windows:

```toml title="pyproject.toml"
[tool.uv]
environments = [
    "sys_platform == 'darwin'",
    "sys_platform == 'linux'",
]
```

Or, to avoid solving for alternative Python implementations:

```toml title="pyproject.toml"
[tool.uv]
environments = [
    "implementation_name == 'cpython'"
]
```

Entries in the `environments` setting must be disjoint (i.e., they must not overlap). For example,
`sys_platform == 'darwin'` and `sys_platform == 'linux'` are disjoint, but
`sys_platform == 'darwin'` and `python_version >= '3.9'` are not, since both could be true at the
same time.

### Required environments

In the Python ecosystem, packages can be published as source distributions, built distributions
(wheels), or both; but to install a package, a built distribution is required. If a package lacks a
built distribution, or lacks a distribution for the current platform or Python version (built
distributions are often platform-specific), uv will attempt to build the package from source, then
install the resulting built distribution.

Some packages (like PyTorch) publish built distributions, but omit a source distribution. Such
packages are _only_ installable on platforms for which a built distribution is available. For
example, if a package publishes built distributions for Linux, but not macOS or Windows, then that
package will _only_ be installable on Linux.

Packages that lack source distributions cause problems for universal resolution, since there will
typically be at least one platform or Python version for which the package is not installable.

By default, uv requires each such package to include at least one wheel that is compatible with the
target Python version. The `required-environments` setting can be used to ensure that the resulting
resolution contains wheels for specific platforms, or fails if no such wheels are available. The
setting accepts a list of
[PEP 508 environment markers](https://packaging.python.org/en/latest/specifications/dependency-specifiers/#environment-markers).

While the `environments` setting _limits_ the set of environments that uv will consider when
resolving dependencies, `required-environments` _expands_ the set of platforms that uv _must_
support when resolving dependencies.

For example, `environments = ["sys_platform == 'darwin'"]` would limit uv to solving for macOS (and
ignoring Linux and Windows). On the other hand,
`required-environments = ["sys_platform == 'darwin'"]` would _require_ that any package without a
source distribution include a wheel for macOS in order to be installable (and would fail if no such
wheel is available).

In practice, `required-environments` can be useful for declaring explicit support for non-latest
platforms, since this often requires backtracking past the latest published versions of those
packages. For example, to guarantee that any built distribution-only packages includes support for
Intel macOS:

```toml title="pyproject.toml"
[tool.uv]
required-environments = [
    "sys_platform == 'darwin' and platform_machine == 'x86_64'"
]
```

## Dependency preferences

If resolution output file exists, i.e., a uv lockfile (`uv.lock`) or a requirements output file
(`requirements.txt`), uv will _prefer_ the dependency versions listed there. Similarly, if
installing a package into a virtual environment, uv will prefer the already installed version if
present. This means that locked or installed versions will not change unless an incompatible version
is requested or an upgrade is explicitly requested with `--upgrade`.

## Resolution strategy

By default, uv tries to use the latest version of each package. For example,
`uv pip install flask>=2.0.0` will install the latest version of Flask, e.g., 3.0.0. If
`flask>=2.0.0` is a dependency of the project, only `flask` 3.0.0 will be used. This is important,
for example, because running tests will not check that the project is actually compatible with its
stated lower bound of `flask` 2.0.0.

With `--resolution lowest`, uv will install the lowest possible version for all dependencies, both
direct and indirect (transitive). Alternatively, `--resolution lowest-direct` will use the lowest
compatible versions for all direct dependencies, while using the latest compatible versions for all
other dependencies. uv will always use the latest versions for build dependencies.

For example, given the following `requirements.in` file:

```python title="requirements.in"
flask>=2.0.0
```

Running `uv pip compile requirements.in` would produce the following `requirements.txt` file:

```python title="requirements.txt"
# This file was autogenerated by uv via the following command:
#    uv pip compile requirements.in
blinker==1.7.0
    # via flask
click==8.1.7
    # via flask
flask==3.0.0
itsdangerous==2.1.2
    # via flask
jinja2==3.1.2
    # via flask
markupsafe==2.1.3
    # via
    #   jinja2
    #   werkzeug
werkzeug==3.0.1
    # via flask
```

However, `uv pip compile --resolution lowest requirements.in` would instead produce:

```python title="requirements.in"
# This file was autogenerated by uv via the following command:
#    uv pip compile requirements.in --resolution lowest
click==7.1.2
    # via flask
flask==2.0.0
itsdangerous==2.0.0
    # via flask
jinja2==3.0.0
    # via flask
markupsafe==2.0.0
    # via jinja2
werkzeug==2.0.0
    # via flask
```

When publishing libraries, it is recommended to separately run tests with `--resolution lowest` or
`--resolution lowest-direct` in continuous integration to ensure compatibility with the declared
lower bounds.

## Pre-release handling

By default, uv will accept pre-release versions during dependency resolution in two cases:

1. If the package is a direct dependency, and its version specifiers include a pre-release specifier
   (e.g., `flask>=2.0.0rc1`).
1. If _all_ published versions of a package are pre-releases.

If dependency resolution fails due to a transitive pre-release, uv will prompt use of
`--prerelease allow` to allow pre-releases for all dependencies.

Alternatively, the transitive dependency can be added as a [constraint](#dependency-constraints) or
direct dependency (i.e. in `requirements.in` or `pyproject.toml`) with a pre-release version
specifier (e.g., `flask>=2.0.0rc1`) to opt-in to pre-release support for that specific dependency.

Pre-releases are
[notoriously difficult](https://pubgrub-rs-guide.netlify.app/limitations/prerelease_versions) to
model, and are a frequent source of bugs in other packaging tools. uv's pre-release handling is
_intentionally_ limited and requires user opt-in for pre-releases to ensure correctness.

For more details, see
[Pre-release compatibility](../pip/compatibility.md#pre-release-compatibility).

## Multi-version resolution

During universal resolution, a package may be listed multiple times with different versions or URLs
within the same lockfile, since different versions may be needed for different platforms or Python
versions.

The `--fork-strategy` setting can be used to control how uv trades off between (1) minimizing the
number of selected versions and (2) selecting the latest-possible version for each platform. The
former leads to greater consistency across platforms, while the latter leads to use of newer package
versions where possible.

By default (`--fork-strategy requires-python`), uv will optimize for selecting the latest version of
each package for each supported Python version, while minimizing the number of selected versions
across platforms.

For example, when resolving `numpy` with a Python requirement of `>=3.8`, uv would select the
following versions:

```txt
numpy==1.24.4 ; python_version == "3.8"
numpy==2.0.2 ; python_version == "3.9"
numpy==2.2.0 ; python_version >= "3.10"
```

This resolution reflects the fact that NumPy 2.2.0 and later require at least Python 3.10, while
earlier versions are compatible with Python 3.8 and 3.9.

Under `--fork-strategy fewest`, uv will instead minimize the number of selected versions for each
package, preferring older versions that are compatible with a wider range of supported Python
versions or platforms.

For example, when in the scenario above, uv would select `numpy==1.24.4` for all Python versions,
rather than upgrading to `numpy==2.0.2` for Python 3.9 and `numpy==2.2.0` for Python 3.10 and later.

## Dependency constraints

Like pip, uv supports constraint files (`--constraint constraints.txt`) which narrow the set of
acceptable versions for the given packages. Constraint files are similar to requirements files, but
being listed as a constraint alone will not cause a package to be included to the resolution.
Instead, constraints only take effect if a requested package is already pulled in as a direct or
transitive dependency. Constraints are useful for reducing the range of available versions for a
transitive dependency. They can also be used to keep a resolution in sync with some other set of
resolved versions, regardless of which packages are overlapping between the two.

## Dependency overrides

Dependency overrides allow bypassing unsuccessful or undesirable resolutions by overriding a
package's declared dependencies. Overrides are a useful last resort for cases in which you _know_
that a dependency is compatible with a certain version of a package, despite the metadata indicating
otherwise.

For example, if a transitive dependency declares the requirement `pydantic>=1.0,<2.0`, but _does_
work with `pydantic>=2.0`, the user can override the declared dependency by including
`pydantic>=1.0,<3` in the overrides, thereby allowing the resolver to choose a newer version of
`pydantic`.

Concretely, if `pydantic>=1.0,<3` is included as an override, uv will ignore all declared
requirements on `pydantic`, replacing them with the override. In the above example, the
`pydantic>=1.0,<2.0` requirement would be ignored completely, and would instead be replaced with
`pydantic>=1.0,<3`.

While constraints can only _reduce_ the set of acceptable versions for a package, overrides can
_expand_ the set of acceptable versions, providing an escape hatch for erroneous upper version
bounds. As with constraints, overrides do not add a dependency on the package and only take effect
if the package is requested in a direct or transitive dependency.

In a `pyproject.toml`, use `tool.uv.override-dependencies` to define a list of overrides. In the
pip-compatible interface, the `--override` option can be used to pass files with the same format as
constraints files.

If multiple overrides are provided for the same package, they must be differentiated with
[markers](#platform-markers). If a package has a dependency with a marker, it is replaced
unconditionally when using overrides — it does not matter if the marker evaluates to true or false.

## Dependency metadata

During resolution, uv needs to resolve the metadata for each package it encounters, in order to
determine its dependencies. This metadata is often available as a static file in the package index;
however, for packages that only provide source distributions, the metadata may not be available
upfront.

In such cases, uv has to build the package to determine its metadata (e.g., by invoking `setup.py`).
This can introduce a performance penalty during resolution. Further, it imposes the requirement that
the package can be built on all platforms, which may not be true.

For example, you may have a package that should only be built and installed on Linux, but doesn't
build successfully on macOS or Windows. While uv can construct a perfectly valid lockfile for this
scenario, doing so would require building the package, which would fail on non-Linux platforms.

The `tool.uv.dependency-metadata` table can be used to provide static metadata for such dependencies
upfront, thereby allowing uv to skip the build step and use the provided metadata instead.

For example, to provide metadata for `chumpy` upfront, include its `dependency-metadata` in the
`pyproject.toml`:

```toml
[[tool.uv.dependency-metadata]]
name = "chumpy"
version = "0.70"
requires-dist = ["numpy>=1.8.1", "scipy>=0.13.0", "six>=1.11.0"]
```

These declarations are intended for cases in which a package does _not_ declare static metadata
upfront, though they are also useful for packages that require disabling build isolation. In such
cases, it may be easier to declare the package metadata upfront, rather than creating a custom build
environment prior to resolving the package.

For example, you can declare the metadata for `flash-attn`, allowing uv to resolve without building
the package from source (which itself requires installing `torch`):

```toml
[project]
name = "project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["flash-attn"]

[tool.uv.sources]
flash-attn = { git = "https://github.com/Dao-AILab/flash-attention", tag = "v2.6.3" }

[[tool.uv.dependency-metadata]]
name = "flash-attn"
version = "2.6.3"
requires-dist = ["torch", "einops"]
```

Like dependency overrides, `tool.uv.dependency-metadata` can also be used for cases in which a
package's metadata is incorrect or incomplete, or when a package is not available in the package
index. While dependency overrides allow overriding the allowed versions of a package globally,
metadata overrides allow overriding the declared metadata of a _specific package_.

!!! note

    The `version` field in `tool.uv.dependency-metadata` is optional for registry-based
    dependencies (when omitted, uv will assume the metadata applies to all versions of the package),
    but _required_ for direct URL dependencies (like Git dependencies).

Entries in the `tool.uv.dependency-metadata` table follow the
[Metadata 2.3](https://packaging.python.org/en/latest/specifications/core-metadata/) specification,
though only `name`, `version`, `requires-dist`, `requires-python`, and `provides-extra` are read by
uv. The `version` field is also considered optional. If omitted, the metadata will be used for all
versions of the specified package.

## Lower bounds

By default, `uv add` adds lower bounds to dependencies and, when using uv to manage projects, uv
will warn if direct dependencies don't have lower bound.

Lower bounds are not critical in the "happy path", but they are important for cases where there are
dependency conflicts. For example, consider a project that requires two packages and those packages
have conflicting dependencies. The resolver needs to check all combinations of all versions within
the constraints for the two packages — if all of them conflict, an error is reported because the
dependencies are not satisfiable. If there are no lower bounds, the resolver can (and often will)
backtrack down to the oldest version of a package. This isn't only problematic because it's slow,
the old version of the package often fails to build, or the resolver can end up picking a version
that's old enough that it doesn't depend on the conflicting package, but also doesn't work with your
code.

Lower bounds are particularly critical when writing a library. It's important to declare the lowest
version for each dependency that your library works with, and to validate that the bounds are
correct — testing with
[`--resolution lowest` or `--resolution lowest-direct`](#resolution-strategy). Otherwise, a user may
receive an old, incompatible version of one of your library's dependencies and the library will fail
with an unexpected error.

## Reproducible resolutions

uv supports an `--exclude-newer` option to limit resolution to distributions published before a
specific date, allowing reproduction of installations regardless of new package releases. The date
may be specified as an [RFC 3339](https://www.rfc-editor.org/rfc/rfc3339.html) timestamp (e.g.,
`2006-12-02T02:07:43Z`) or a local date in the same format (e.g., `2006-12-02`) in your system's
configured time zone.

Note the package index must support the `upload-time` field as specified in
[`PEP 700`](https://peps.python.org/pep-0700/). If the field is not present for a given
distribution, the distribution will be treated as unavailable. PyPI provides `upload-time` for all
packages.

To ensure reproducibility, messages for unsatisfiable resolutions will not mention that
distributions were excluded due to the `--exclude-newer` flag — newer distributions will be treated
as if they do not exist.

!!! note

    The `--exclude-newer` option is only applied to packages that are read from a registry (as opposed to, e.g., Git
    dependencies). Further, when using the `uv pip` interface, uv will not downgrade previously installed packages
    unless the `--reinstall` flag is provided, in which case uv will perform a new resolution.

## Source distribution

[PEP 625](https://peps.python.org/pep-0625/) specifies that packages must distribute source
distributions as gzip tarball (`.tar.gz`) archives. Prior to this specification, other archive
formats, which need to be supported for backward compatibility, were also allowed. uv supports
reading and extracting archives in the following formats:

- gzip tarball (`.tar.gz`, `.tgz`)
- bzip2 tarball (`.tar.bz2`, `.tbz`)
- xz tarball (`.tar.xz`, `.txz`)
- zstd tarball (`.tar.zst`)
- lzip tarball (`.tar.lz`)
- lzma tarball (`.tar.lzma`)
- zip (`.zip`)

## Learn more

For more details about the internals of the resolver, see the
[resolver reference](../reference/resolver-internals.md) documentation.

## Lockfile versioning

The `uv.lock` file uses a versioned schema. The schema version is included in the `version` field of
the lockfile.

Any given version of uv can read and write lockfiles with the same schema version, but will reject
lockfiles with a greater schema version. For example, if your uv version supports schema v1,
`uv lock` will error if it encounters an existing lockfile with schema v2.

uv versions that support schema v2 _may_ be able to read lockfiles with schema v1 if the schema
update was backwards-compatible. However, this is not guaranteed, and uv may exit with an error if
it encounters a lockfile with an outdated schema version.

The schema version is considered part of the public API, and so is only bumped in minor releases, as
a breaking change (see [Versioning](../reference/policies/versioning.md)). As such, all uv patch
versions within a given minor uv release are guaranteed to have full lockfile compatibility. In
other words, lockfiles may only be rejected across minor releases.

# Caching System

# Caching

## Dependency caching

uv uses aggressive caching to avoid re-downloading (and re-building) dependencies that have already
been accessed in prior runs.

The specifics of uv's caching semantics vary based on the nature of the dependency:

- **For registry dependencies** (like those downloaded from PyPI), uv respects HTTP caching headers.
- **For direct URL dependencies**, uv respects HTTP caching headers, and also caches based on the
  URL itself.
- **For Git dependencies**, uv caches based on the fully-resolved Git commit hash. As such,
  `uv pip compile` will pin Git dependencies to a specific commit hash when writing the resolved
  dependency set.
- **For local dependencies**, uv caches based on the last-modified time of the source archive (i.e.,
  the local `.whl` or `.tar.gz` file). For directories, uv caches based on the last-modified time of
  the `pyproject.toml`, `setup.py`, or `setup.cfg` file.

If you're running into caching issues, uv includes a few escape hatches:

- To force uv to revalidate cached data for all dependencies, pass `--refresh` to any command (e.g.,
  `uv sync --refresh` or `uv pip install --refresh ...`).
- To force uv to revalidate cached data for a specific dependency pass `--refresh-package` to any
  command (e.g., `uv sync --refresh-package flask` or `uv pip install --refresh-package flask ...`).
- To force uv to ignore existing installed versions, pass `--reinstall` to any installation command
  (e.g., `uv sync --reinstall` or `uv pip install --reinstall ...`).

As a special case, uv will always rebuild and reinstall any local directory dependencies passed
explicitly on the command-line (e.g., `uv pip install .`).

## Dynamic metadata

By default, uv will _only_ rebuild and reinstall local directory dependencies (e.g., editables) if
the `pyproject.toml`, `setup.py`, or `setup.cfg` file in the directory root has changed, or if a
`src` directory is added or removed. This is a heuristic and, in some cases, may lead to fewer
re-installs than desired.

To incorporate additional information into the cache key for a given package, you can add cache key
entries under [`tool.uv.cache-keys`](https://docs.astral.sh/uv/reference/settings/#cache-keys),
which covers both file paths and Git commit hashes. Setting
[`tool.uv.cache-keys`](https://docs.astral.sh/uv/reference/settings/#cache-keys) will replace
defaults, so any necessary files (like `pyproject.toml`) should still be included in the
user-defined cache keys.

For example, if a project specifies dependencies in `pyproject.toml` but uses
[`setuptools-scm`](https://pypi.org/project/setuptools-scm/) to manage its version, and should thus
be rebuilt whenever the commit hash or dependencies change, you can add the following to the
project's `pyproject.toml`:

```toml title="pyproject.toml"
[tool.uv]
cache-keys = [{ file = "pyproject.toml" }, { git = { commit = true } }]
```

If your dynamic metadata incorporates information from the set of Git tags, you can expand the cache
key to include the tags:

```toml title="pyproject.toml"
[tool.uv]
cache-keys = [{ file = "pyproject.toml" }, { git = { commit = true, tags = true } }]
```

Similarly, if a project reads from a `requirements.txt` to populate its dependencies, you can add
the following to the project's `pyproject.toml`:

```toml title="pyproject.toml"
[tool.uv]
cache-keys = [{ file = "pyproject.toml" }, { file = "requirements.txt" }]
```

Globs are supported for `file` keys, following the syntax of the
[`glob`](https://docs.rs/glob/0.3.1/glob/struct.Pattern.html) crate. For example, to invalidate the
cache whenever a `.toml` file in the project directory or any of its subdirectories is modified, use
the following:

```toml title="pyproject.toml"
[tool.uv]
cache-keys = [{ file = "**/*.toml" }]
```

!!! note

    The use of globs can be expensive, as uv may need to walk the filesystem to determine whether any files have changed.
    This may, in turn, requiring traversal of large or deeply nested directories.

Similarly, if a project relies on an environment variable, you can add the following to the
project's `pyproject.toml` to invalidate the cache whenever the environment variable changes:

```toml title="pyproject.toml"
[tool.uv]
cache-keys = [{ file = "pyproject.toml" }, { env = "MY_ENV_VAR" }]
```

Finally, to invalidate a project whenever a specific directory (like `src`) is created or removed,
add the following to the project's `pyproject.toml`:

```toml title="pyproject.toml"
[tool.uv]
cache-keys = [{ file = "pyproject.toml" }, { dir = "src" }]
```

Note that the `dir` key will only track changes to the directory itself, and not arbitrary changes
within the directory.

As an escape hatch, if a project uses `dynamic` metadata that isn't covered by `tool.uv.cache-keys`,
you can instruct uv to _always_ rebuild and reinstall it by adding the project to the
`tool.uv.reinstall-package` list:

```toml title="pyproject.toml"
[tool.uv]
reinstall-package = ["my-package"]
```

This will force uv to rebuild and reinstall `my-package` on every run, regardless of whether the
package's `pyproject.toml`, `setup.py`, or `setup.cfg` file has changed.

## Cache safety

It's safe to run multiple uv commands concurrently, even against the same virtual environment. uv's
cache is designed to be thread-safe and append-only, and thus robust to multiple concurrent readers
and writers. uv applies a file-based lock to the target virtual environment when installing, to
avoid concurrent modifications across processes.

Note that it's _not_ safe to modify the uv cache (e.g., `uv cache clean`) while other uv commands
are running, and _never_ safe to modify the cache directly (e.g., by removing a file or directory).

## Clearing the cache

uv provides a few different mechanisms for removing entries from the cache:

- `uv cache clean` removes _all_ cache entries from the cache directory, clearing it out entirely.
- `uv cache clean ruff` removes all cache entries for the `ruff` package, useful for invalidating
  the cache for a single or finite set of packages.
- `uv cache prune` removes all _unused_ cache entries. For example, the cache directory may contain
  entries created in previous uv versions that are no longer necessary and can be safely removed.
  `uv cache prune` is safe to run periodically, to keep the cache directory clean.

## Caching in continuous integration

It's common to cache package installation artifacts in continuous integration environments (like
GitHub Actions or GitLab CI) to speed up subsequent runs.

By default, uv caches both the wheels that it builds from source and the pre-built wheels that it
downloads directly, to enable high-performance package installation.

However, in continuous integration environments, persisting pre-built wheels may be undesirable.
With uv, it turns out that it's often faster to _omit_ pre-built wheels from the cache (and instead
re-download them from the registry on each run). On the other hand, caching wheels that are built
from source tends to be worthwhile, since the wheel building process can be expensive, especially
for extension modules.

To support this caching strategy, uv provides a `uv cache prune --ci` command, which removes all
pre-built wheels and unzipped source distributions from the cache, but retains any wheels that were
built from source. We recommend running `uv cache prune --ci` at the end of your continuous
integration job to ensure maximum cache efficiency. For an example, see the
[GitHub integration guide](../guides/integration/github.md#caching).

## Cache directory

uv determines the cache directory according to, in order:

1. A temporary cache directory, if `--no-cache` was requested.
2. The specific cache directory specified via `--cache-dir`, `UV_CACHE_DIR`, or
   [`tool.uv.cache-dir`](../reference/settings.md#cache-dir).
3. A system-appropriate cache directory, e.g., `$XDG_CACHE_HOME/uv` or `$HOME/.cache/uv` on Unix and
   `%LOCALAPPDATA%\uv\cache` on Windows

!!! note

    uv _always_ requires a cache directory. When `--no-cache` is requested, uv will still use
    a temporary cache for sharing data within that single invocation.

    In most cases, `--refresh` should be used instead of `--no-cache` — as it will update the cache
    for subsequent operations but not read from the cache.

It is important for performance for the cache directory to be located on the same file system as the
Python environment uv is operating on. Otherwise, uv will not be able to link files from the cache
into the environment and will instead need to fallback to slow copy operations.

## Cache versioning

The uv cache is composed of a number of buckets (e.g., a bucket for wheels, a bucket for source
distributions, a bucket for Git repositories, and so on). Each bucket is versioned, such that if a
release contains a breaking change to the cache format, uv will not attempt to read from or write to
an incompatible cache bucket.

For example, uv 0.4.13 included a breaking change to the core metadata bucket. As such, the bucket
version was increased from v12 to v13. Within a cache version, changes are guaranteed to be both
forwards- and backwards-compatible.

Since changes in the cache format are accompanied by changes in the cache version, multiple versions
of uv can safely read and write to the same cache directory. However, if the cache version changed
between a given pair of uv releases, then those releases may not be able to share the same
underlying cache entries.

For example, it's safe to use a single shared cache for uv 0.4.12 and uv 0.4.13, though the cache
itself may contain duplicate entries in the core metadata bucket due to the change in cache version.

# The Pip Interface

# The pip interface

uv provides a drop-in replacement for common `pip`, `pip-tools`, and `virtualenv` commands. These
commands work directly with the virtual environment, in contrast to uv's primary interfaces where
the virtual environment is managed automatically. The `uv pip` interface exposes the speed and
functionality of uv to power users and projects that are not ready to transition away from `pip` and
`pip-tools`.

The following sections discuss the basics of using `uv pip`:

- [Creating and using environments](./environments.md)
- [Installing and managing packages](./packages.md)
- [Inspecting environments and packages](./inspection.md)
- [Declaring package dependencies](./dependencies.md)
- [Locking and syncing environments](./compile.md)

Please note these commands do not _exactly_ implement the interfaces and behavior of the tools they
are based on. The further you stray from common workflows, the more likely you are to encounter
differences. Consult the [pip-compatibility guide](./compatibility.md) for details.

!!! important

    uv does not rely on or invoke pip. The pip interface is named as such to highlight its dedicated
    purpose of providing low-level commands that match pip's interface and to separate it from the
    rest of uv's commands which operate at a higher level of abstraction.

# Script Management

---
title: Running scripts
description:
  A guide to using uv to run Python scripts, including support for inline dependency metadata,
  reproducible scripts, and more.
---

# Running scripts

A Python script is a file intended for standalone execution, e.g., with `python <script>.py`. Using
uv to execute scripts ensures that script dependencies are managed without manually managing
environments.

!!! note

    If you are not familiar with Python environments: every Python installation has an environment
    that packages can be installed in. Typically, creating [_virtual_ environments](https://docs.python.org/3/library/venv.html) is recommended to
    isolate packages required by each script. uv automatically manages virtual environments for you
    and prefers a [declarative](#declaring-script-dependencies) approach to dependencies.

## Running a script without dependencies

If your script has no dependencies, you can execute it with `uv run`:

```python title="example.py"
print("Hello world")
```

```console
$ uv run example.py
Hello world
```

<!-- TODO(zanieb): Once we have a `python` shim, note you can execute it with `python` here -->

Similarly, if your script depends on a module in the standard library, there's nothing more to do:

```python title="example.py"
import os

print(os.path.expanduser("~"))
```

```console
$ uv run example.py
/Users/astral
```

Arguments may be provided to the script:

```python title="example.py"
import sys

print(" ".join(sys.argv[1:]))
```

```console
$ uv run example.py test
test

$ uv run example.py hello world!
hello world!
```

Additionally, your script can be read directly from stdin:

```console
$ echo 'print("hello world!")' | uv run -
```

Or, if your shell supports [here-documents](https://en.wikipedia.org/wiki/Here_document):

```bash
uv run - <<EOF
print("hello world!")
EOF
```

Note that if you use `uv run` in a _project_, i.e., a directory with a `pyproject.toml`, it will
install the current project before running the script. If your script does not depend on the
project, use the `--no-project` flag to skip this:

```console
$ # Note: the `--no-project` flag must be provided _before_ the script name.
$ uv run --no-project example.py
```

See the [projects guide](./projects.md) for more details on working in projects.

## Running a script with dependencies

When your script requires other packages, they must be installed into the environment that the
script runs in. uv prefers to create these environments on-demand instead of using a long-lived
virtual environment with manually managed dependencies. This requires explicit declaration of
dependencies that are required for the script. Generally, it's recommended to use a
[project](./projects.md) or [inline metadata](#declaring-script-dependencies) to declare
dependencies, but uv supports requesting dependencies per invocation as well.

For example, the following script requires `rich`.

```python title="example.py"
import time
from rich.progress import track

for i in track(range(20), description="For example:"):
    time.sleep(0.05)
```

If executed without specifying a dependency, this script will fail:

```console
$ uv run --no-project example.py
Traceback (most recent call last):
  File "/Users/astral/example.py", line 2, in <module>
    from rich.progress import track
ModuleNotFoundError: No module named 'rich'
```

Request the dependency using the `--with` option:

```console
$ uv run --with rich example.py
For example: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
```

Constraints can be added to the requested dependency if specific versions are needed:

```console
$ uv run --with 'rich>12,<13' example.py
```

Multiple dependencies can be requested by repeating with `--with` option.

Note that if `uv run` is used in a _project_, these dependencies will be included _in addition_ to
the project's dependencies. To opt-out of this behavior, use the `--no-project` flag.

## Creating a Python script

Python recently added a standard format for
[inline script metadata](https://packaging.python.org/en/latest/specifications/inline-script-metadata/#inline-script-metadata).
It allows for selecting Python versions and defining dependencies. Use `uv init --script` to
initialize scripts with the inline metadata:

```console
$ uv init --script example.py --python 3.12
```

## Declaring script dependencies

The inline metadata format allows the dependencies for a script to be declared in the script itself.

uv supports adding and updating inline script metadata for you. Use `uv add --script` to declare the
dependencies for the script:

```console
$ uv add --script example.py 'requests<3' 'rich'
```

This will add a `script` section at the top of the script declaring the dependencies using TOML:

```python title="example.py"
# /// script
# dependencies = [
#   "requests<3",
#   "rich",
# ]
# ///

import requests
from rich.pretty import pprint

resp = requests.get("https://peps.python.org/api/peps.json")
data = resp.json()
pprint([(k, v["title"]) for k, v in data.items()][:10])
```

uv will automatically create an environment with the dependencies necessary to run the script, e.g.:

```console
$ uv run example.py
[
│   ('1', 'PEP Purpose and Guidelines'),
│   ('2', 'Procedure for Adding New Modules'),
│   ('3', 'Guidelines for Handling Bug Reports'),
│   ('4', 'Deprecation of Standard Modules'),
│   ('5', 'Guidelines for Language Evolution'),
│   ('6', 'Bug Fix Releases'),
│   ('7', 'Style Guide for C Code'),
│   ('8', 'Style Guide for Python Code'),
│   ('9', 'Sample Plaintext PEP Template'),
│   ('10', 'Voting Guidelines')
]
```

!!! important

    When using inline script metadata, even if `uv run` is [used in a _project_](../concepts/projects/run.md), the project's dependencies will be ignored. The `--no-project` flag is not required.

uv also respects Python version requirements:

```python title="example.py"
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

# Use some syntax added in Python 3.12
type Point = tuple[float, float]
print(Point)
```

!!! note

    The `dependencies` field must be provided even if empty.

`uv run` will search for and use the required Python version. The Python version will download if it
is not installed — see the documentation on [Python versions](../concepts/python-versions.md) for
more details.

## Using a shebang to create an executable file

A shebang can be added to make a script executable without using `uv run` — this makes it easy to
run scripts that are on your `PATH` or in the current folder.

For example, create a file called `greet` with the following contents

```python title="greet"
#!/usr/bin/env -S uv run --script

print("Hello, world!")
```

Ensure that your script is executable, e.g., with `chmod +x greet`, then run the script:

```console
$ ./greet
Hello, world!
```

Declaration of dependencies is also supported in this context, for example:

```python title="example"
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
import httpx

print(httpx.get("https://example.com"))
```

## Using alternative package indexes

If you wish to use an alternative [package index](../configuration/indexes.md) to resolve
dependencies, you can provide the index with the `--index` option:

```console
$ uv add --index "https://example.com/simple" --script example.py 'requests<3' 'rich'
```

This will include the package data in the inline metadata:

```python
# [[tool.uv.index]]
# url = "https://example.com/simple"
```

If you require authentication to access the package index, then please refer to the
[package index](../configuration/indexes.md) documentation.

## Locking dependencies

uv supports locking dependencies for PEP 723 scripts using the `uv.lock` file format. Unlike with
projects, scripts must be explicitly locked using `uv lock`:

```console
$ uv lock --script example.py
```

Running `uv lock --script` will create a `.lock` file adjacent to the script (e.g.,
`example.py.lock`).

Once locked, subsequent operations like `uv run --script`, `uv add --script`, `uv export --script`,
and `uv tree --script` will reuse the locked dependencies, updating the lockfile if necessary.

If no such lockfile is present, commands like `uv export --script` will still function as expected,
but will not create a lockfile.

## Improving reproducibility

In addition to locking dependencies, uv supports an `exclude-newer` field in the `tool.uv` section
of inline script metadata to limit uv to only considering distributions released before a specific
date. This is useful for improving the reproducibility of your script when run at a later point in
time.

The date must be specified as an [RFC 3339](https://www.rfc-editor.org/rfc/rfc3339.html) timestamp
(e.g., `2006-12-02T02:07:43Z`).

```python title="example.py"
# /// script
# dependencies = [
#   "requests",
# ]
# [tool.uv]
# exclude-newer = "2023-10-16T00:00:00Z"
# ///

import requests

print(requests.__version__)
```

## Using different Python versions

uv allows arbitrary Python versions to be requested on each script invocation, for example:

```python title="example.py"
import sys

print(".".join(map(str, sys.version_info[:3])))
```

```console
$ # Use the default Python version, may differ on your machine
$ uv run example.py
3.12.6
```

```console
$ # Use a specific Python version
$ uv run --python 3.10 example.py
3.10.15
```

See the [Python version request](../concepts/python-versions.md#requesting-a-version) documentation
for more details on requesting Python versions.

## Using GUI scripts

On Windows `uv` will run your script ending with `.pyw` extension using `pythonw`:

```python title="example.pyw"
from tkinter import Tk, ttk

root = Tk()
root.title("uv")
frm = ttk.Frame(root, padding=10)
frm.grid()
ttk.Label(frm, text="Hello World").grid(column=0, row=0)
root.mainloop()
```

```console
PS> uv run example.pyw
```

![Run Result](../assets/uv_gui_script_hello_world.png){: style="height:50px;width:150px"}

Similarly, it works with dependencies as well:

```python title="example_pyqt.pyw"
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout

app = QApplication(sys.argv)
widget = QWidget()
grid = QGridLayout()

text_label = QLabel()
text_label.setText("Hello World!")
grid.addWidget(text_label)

widget.setLayout(grid)
widget.setGeometry(100, 100, 200, 50)
widget.setWindowTitle("uv")
widget.show()
sys.exit(app.exec_())
```

```console
PS> uv run --with PyQt5 example_pyqt.pyw
```

![Run Result](../assets/uv_gui_script_hello_world_pyqt.png){: style="height:50px;width:150px"}

## Next steps

To learn more about `uv run`, see the [command reference](../reference/cli.md#uv-run).

Or, read on to learn how to [run and install tools](./tools.md) with uv.

# Project Management

---
title: Working on projects
description:
  A guide to using uv to create and manage Python projects, including adding dependencies, running
  commands, and building publishable distributions.
---

# Working on projects

uv supports managing Python projects, which define their dependencies in a `pyproject.toml` file.

## Creating a new project

You can create a new Python project using the `uv init` command:

```console
$ uv init hello-world
$ cd hello-world
```

Alternatively, you can initialize a project in the working directory:

```console
$ mkdir hello-world
$ cd hello-world
$ uv init
```

uv will create the following files:

```text
.
├── .python-version
├── README.md
├── main.py
└── pyproject.toml
```

The `main.py` file contains a simple "Hello world" program. Try it out with `uv run`:

```console
$ uv run main.py
Hello from hello-world!
```

## Project structure

A project consists of a few important parts that work together and allow uv to manage your project.
In addition to the files created by `uv init`, uv will create a virtual environment and `uv.lock`
file in the root of your project the first time you run a project command, i.e., `uv run`,
`uv sync`, or `uv lock`.

A complete listing would look like:

```text
.
├── .venv
│   ├── bin
│   ├── lib
│   └── pyvenv.cfg
├── .python-version
├── README.md
├── main.py
├── pyproject.toml
└── uv.lock
```

### `pyproject.toml`

The `pyproject.toml` contains metadata about your project:

```toml title="pyproject.toml"
[project]
name = "hello-world"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
dependencies = []
```

You'll use this file to specify dependencies, as well as details about the project such as its
description or license. You can edit this file manually, or use commands like `uv add` and
`uv remove` to manage your project from the terminal.

!!! tip

    See the official [`pyproject.toml` guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
    for more details on getting started with the `pyproject.toml` format.

You'll also use this file to specify uv [configuration options](../configuration/files.md) in a
[`[tool.uv]`](../reference/settings.md) section.

### `.python-version`

The `.python-version` file contains the project's default Python version. This file tells uv which
Python version to use when creating the project's virtual environment.

### `.venv`

The `.venv` folder contains your project's virtual environment, a Python environment that is
isolated from the rest of your system. This is where uv will install your project's dependencies.

See the [project environment](../concepts/projects/layout.md#the-project-environment) documentation
for more details.

### `uv.lock`

`uv.lock` is a cross-platform lockfile that contains exact information about your project's
dependencies. Unlike the `pyproject.toml` which is used to specify the broad requirements of your
project, the lockfile contains the exact resolved versions that are installed in the project
environment. This file should be checked into version control, allowing for consistent and
reproducible installations across machines.

`uv.lock` is a human-readable TOML file but is managed by uv and should not be edited manually.

See the [lockfile](../concepts/projects/layout.md#the-lockfile) documentation for more details.

## Managing dependencies

You can add dependencies to your `pyproject.toml` with the `uv add` command. This will also update
the lockfile and project environment:

```console
$ uv add requests
```

You can also specify version constraints or alternative sources:

```console
$ # Specify a version constraint
$ uv add 'requests==2.31.0'

$ # Add a git dependency
$ uv add git+https://github.com/psf/requests
```

If you're migrating from a `requirements.txt` file, you can use `uv add` with the `-r` flag to add
all dependencies from the file:

```console
$ # Add all dependencies from `requirements.txt`.
$ uv add -r requirements.txt -c constraints.txt
```

To remove a package, you can use `uv remove`:

```console
$ uv remove requests
```

To upgrade a package, run `uv lock` with the `--upgrade-package` flag:

```console
$ uv lock --upgrade-package requests
```

The `--upgrade-package` flag will attempt to update the specified package to the latest compatible
version, while keeping the rest of the lockfile intact.

See the documentation on [managing dependencies](../concepts/projects/dependencies.md) for more
details.

## Running commands

`uv run` can be used to run arbitrary scripts or commands in your project environment.

Prior to every `uv run` invocation, uv will verify that the lockfile is up-to-date with the
`pyproject.toml`, and that the environment is up-to-date with the lockfile, keeping your project
in-sync without the need for manual intervention. `uv run` guarantees that your command is run in a
consistent, locked environment.

For example, to use `flask`:

```console
$ uv add flask
$ uv run -- flask run -p 3000
```

Or, to run a script:

```python title="example.py"
# Require a project dependency
import flask

print("hello world")
```

```console
$ uv run example.py
```

Alternatively, you can use `uv sync` to manually update the environment then activate it before
executing a command:

=== "macOS and Linux"

    ```console
    $ uv sync
    $ source .venv/bin/activate
    $ flask run -p 3000
    $ python example.py
    ```

=== "Windows"

    ```pwsh-session
    PS> uv sync
    PS> .venv\Scripts\activate
    PS> flask run -p 3000
    PS> python example.py
    ```

!!! note

    The virtual environment must be active to run scripts and commands in the project without `uv run`. Virtual environment activation differs per shell and platform.

See the documentation on [running commands and scripts](../concepts/projects/run.md) in projects for
more details.

## Building distributions

`uv build` can be used to build source distributions and binary distributions (wheel) for your
project.

By default, `uv build` will build the project in the current directory, and place the built
artifacts in a `dist/` subdirectory:

```console
$ uv build
$ ls dist/
hello-world-0.1.0-py3-none-any.whl
hello-world-0.1.0.tar.gz
```

See the documentation on [building projects](../concepts/projects/build.md) for more details.

## Next steps

To learn more about working on projects with uv, see the
[projects concept](../concepts/projects/index.md) page and the
[command reference](../reference/cli.md#uv).

Or, read on to learn how to [build and publish your project to a package index](./package.md).

# Additional Guides

# Guides overview

Check out one of the core guides to get started:

- [Installing Python versions](./install-python.md)
- [Running scripts and declaring dependencies](./scripts.md)
- [Running and installing applications as tools](./tools.md)
- [Creating and working on projects](./projects.md)
- [Building and publishing packages](./package.md)
- [Integrate uv with other software, e.g., Docker, GitHub, PyTorch, and more](./integration/index.md)

Or, explore the [concept documentation](../concepts/index.md) for comprehensive breakdown of each
feature.
