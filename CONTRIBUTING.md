# How to Contribute 💜

Since you landed on this part of the repository, we want to first of all say thank you! 💜  
Contributions from the community are essential to improving Pruna, we appreciate your effort in making the repository better for everyone!

Please make sure to review and adhere to the [Pruna Code of Conduct](https://github.com/PrunaAI/pruna/blob/main/CODE_OF_CONDUCT.md) before contributing to Pruna.  
Any violations will be handled accordingly and result in a ban from the Pruna community and associated platforms.  
Contributions that do not adhere to the code of conduct will be ignored.

There are various ways you can contribute:

- Have a question? Discuss with us on [Discord](https://discord.gg/Tun8YgzxZ9) or check out the [FAQ](https://docs.pruna.ai/en/stable/docs_pruna/faq.html)
- Have an idea for a new tutorial? Open an issue with a [feature request](https://docs.pruna.ai/en/stable/docs_pruna/contributions/opening_an_issue.html#feature-request) or chat with us on [Discord](https://discord.gg/Tun8YgzxZ9)
- Found a bug? Open an issue with a [bug report](https://docs.pruna.ai/en/stable/docs_pruna/contributions/opening_an_issue.html#bug-report)
- Want a new feature? Open an issue with a [feature request](https://docs.pruna.ai/en/stable/docs_pruna/contributions/opening_an_issue.html#feature-request)
- Have a new algorithm to add? Check out the [guide on adding algorithms](https://docs.pruna.ai/en/stable/docs_pruna/contributions/adding_algorithm.html)
- Have a new metric to add? Check out the [guide on adding metrics](https://docs.pruna.ai/en/stable/docs_pruna/contributions/adding_metric.html)
- Have a new dataset to add? Check out the [guide on adding datasets](https://docs.pruna.ai/en/stable/docs_pruna/contributions/adding_dataset.html)

---

## Setup

If you want to contribute to Pruna with a Pull Request, you can do so by following these steps.
If it is your very first time contributing to an open source project, we recommend to start with [this guide](https://opensource.guide/how-to-contribute/) for some generally helpful tips.

### 1. Clone the repository

First, fork the repository by navigating to the original [pruna repository](https://github.com/PrunaAI/pruna) on GitHub and clicking the **Fork** button at the top-right. This creates a copy of the repository in your own GitHub account. Then, clone the forked repository from your account to your local machine and change into its directory:

```bash
git clone https://github.com/your_username/pruna.git
cd pruna
```

To keep your fork up to date with the original repository, add the upstream remote:

```bash
git remote add upstream https://github.com/PrunaAI/pruna.git
```

Always work on a new branch rather than the main branch. You can create a new branch for your feature or fix:

```bash
git switch -c feat/new-feature
```

### 2. Installation

You can now set up a virtual environment of your choice and install the dependencies by running the following command:

```bash
pip install -e .
pip install -e '.[dev]'
pip install -e '.[tests]'
```

You can then also install the pre-commit hooks with:

```bash
pre-commit install
```

### 3. Develop your contribution

You are now ready to work on your contribution. Check out a branch on your forked repository and start coding! When committing your changes, we recommend following the [Conventional Commit Guidelines](https://www.conventionalcommits.org/en/v1.0.0/).

```bash
git switch -c feat/new-feature
git add .
git commit -m "feat: new amazing feature setup"
git push origin feat/new-feature
```

Make sure to develop your contribution in a way that is well documented, concise, and easy to maintain. We will do our best to have your contribution integrated and maintained into Pruna but reserve the right to reject contributions that we do not feel are in the best interest of the project.

### 4. Run the tests

We have a comprehensive test suite that is designed to catch potential issues before they are merged into Pruna. When you make a contribution, it is highly recommended to not only run the existing tests but also to add new tests that cover your contribution.

You can run the tests by running the following command:

```bash
pytest
```

If you want to run only the tests with a specific marker, e.g. fast CPU tests, you can do so by running:

```bash
pytest -m "cpu and not slow"
```

### 5. Create a Pull Request

Once you have made your changes and tested them, you can create a Pull Request. We will then review your Pull Request and get back to you as soon as possible. If there are any questions along the way, please do not hesitate to reach out on [Discord](https://discord.gg/Tun8YgzxZ9).