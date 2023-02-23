# Contributing to `trlX`

Looking to improve `trlX`? Thanks for considering!

There are many ways to contribute, from writing tutorials in [Colab notebooks](https://colab.research.google.com) to improving the project's [documentation](https://trlx.readthedocs.io), submitting bug reports and feature requests, or even implementing new features themselves. See the outstanding [issues](https://github.com/CarperAI/trlx/issues) for ideas on where to begin.

Here are some guidelines to help you get started 🚀.

## Submitting a bug report or a feature request¶

To submit a bug report or a feature request, please open an [issue](https://github.com/CarperAI/trlx/issues) by clicking on the `New Issue` button and selecting the respective issue template. Make sure to fill out all the required information and provide as much detail as possible. For bug reports, this means including a minimal code example that reproduces the bug, and for feature requests, it means providing a clear and detailed description of the feature you would like to see implemented.

## Submitting code

> **Note**: Make sure to first search through the [issue tracker](https://github.com/CarperAI/trlx/issues) and [PR list](https://github.com/CarperAI/trlx/pulls) to avoid duplicating work. If you want to work on a non-trivial feature, we highly recommended that you first open an issue in the [issue tracker](https://github.com/CarperAI/trlx/issues) to get feedback from core developers.

Follow these steps to start contributing code:

1. Create your own [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo#forking-a-repository) of the repository and clone it to your local machine.
    ```bash
    git clone https://github.com/<YOUR-USERNAME>/trlx.git
    cd trlx
    git remote add upstream https://github.com/CarperAI/trlx.git
    ```
2. Create a new branch for your changes and give it a concise name that reflects your contribution.
    ```bash
    git checkout -b <BRANCH-NAME>
    ```
2. Install the development dependencies in a Python environment.
    ```bash
    pip install -e ".[dev]"
    pre-commit install
    ```
4. Implement your changes. Make small, independent, and well documented commits along the way (check out [these](https://cbea.ms/git-commit/) tips).
5. Add unit tests whenever appropriate and ensure that the tests pass. To run the entire test suite, use the following command from within the project root directory.
    ```bash
    pytest
    ```
    For changes with minimal project scope (e.g. a simple bug fix), you might want to run the unit tests for just a specific test file instead:
    ```bash
    pytest -vv -k "<TEST-FILE-NAME>"
    ```
5. Commit your final changes. Our `pre-commit` hooks will automatically run before each commit and will prevent you from committing code that does not pass our style and linter checks. They'll also automatically format your code! To run these manually, use the following command:
    ```bash
    pre-commit run --all-files
    ```

6. Push the changes to your fork.

Finally ... 🥁 ... Create a [pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request) to the `trlX` repository! Make sure to include a description of your changes and link to any relevant issues.

> __Tip__: If you're looking to introduce an experimental feature, we suggest testing the behavior of your proposed feature on some of the existing [examples](https://github.com/CarperAI/trlx/tree/master/examples), such as [random walks](https://github.com/CarperAI/trlx/blob/master/examples/randomwalks). This will help you get a better sense of how the feature would work in practice and will also help you identify any potential flaws in the implementation.

## Asking questions

Have a question? Rather than opening an issue, you can readily chat with the core team on our [Discord server](https://discord.gg/canadagoose).

## Code of conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://github.com/CarperAI/trlx/blob/master/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## License

By contributing, you agree that your contributions will be licensed under its MIT License.

# Thank you for your contribution 🐠!
