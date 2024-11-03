# Project Guidelines

## Cloning the Repository

Start by cloning the repository to your local machine. You can do this by running the following command in your terminal:

```bash
git clone https://github.com/StephanAkkerman/FluentAI
```

Install the required dev dependencies by running the following command:

```bash
pip install -e .[dev]
```

## Code Formatting and Linting

This project uses [Black](https://black.readthedocs.io/en/stable/index.html) with default values for automatic code formatting, along with [ruff](https://docs.astral.sh/ruff/). We also use [NumPy-style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html) for documenting the code. As part of the checks on pull requests, it is checked whether the code still adheres to the code style. To ensure you don't need to worry about formatting and linting when contributing, it is recommended to set up the following:

- Integration in VS Code:
  - For [Black](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
  - For [ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)
  - For automated [NumPy-style docstrings](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)
    - You need to change the `docstringFormat` setting to NumPy (see below)
  - Optional: I suggest to install [Mogami](https://marketplace.visualstudio.com/items?itemName=ninoseki.vscode-mogami) to keep your dependencies up to date

### VS Code Settings

To enable organzing the import on save we need to change the settings.
You can find this by pressing `F1` or `Ctrl+Shift+P` in VS code and typing `Preferences: Open User Settings (JSON)`.
Paste the contents of the following JSON document in there.

```json
{
  "autoDocstring.docstringFormat": "numpy",
  "[python]": {
    "editor.codeActionsOnSave": {
      "source.organizeImports": "always"
    }
  }
}
```

## Issue Reporting Guidelines

To maintain consistency and clarity when reporting issues, please use the following template:

### Issue Template

1. Title: [Clear and concise issue title]

2. Description:

   - Problem:
     [Briefly describe the issue or gap that needs to be addressed.]

   - Solution:
     [Outline the proposed action or changes to resolve the problem. If none yet, leave blank]

   - Prerequisites:
     [List any requirements or dependencies needed before starting.]

3. Tasks:

   - Task 1
   - Task 2
   - Task 3
   - etc.

### Guidelines for Submitting Issues

1. Search Existing Issues:
   Before creating a new issue, please check if it has already been reported to avoid duplicates.

2. Use the Template:
   Fill out all sections of the issue template to provide comprehensive information.

3. Be Clear and Concise:
   Use straightforward language and provide as much relevant detail as possible.

4. Label Appropriately:
   Assign appropriate labels (e.g., bug, enhancement, documentation) to categorize the issue.

## Submitting Pull Requests

When submitting a pull request, please follow these guidelines:

1. Fork the repository and create your branch from `main`.
2. Ensure your branch is up to date with the latest `main` branch.
3. Write clear and descriptive commit messages.
4. Include a detailed description of your changes in the pull request and the issues that the PR closes.
5. Ensure that your code passes all tests and linter checks.

## Pull Request Merging Policy

To ensure a smooth and transparent merging process, we have established the following policy for merging pull requests:

1. **Review and Approval**:

   - Pull requests must be reviewed and approved by at least one project maintainer.
   - Reviews from other contributors are encouraged but not required.

2. **Passing Checks**:

   - All pull requests must pass continuous integration (CI) checks, including tests and linters.
   - Ensure there are no merge conflicts with the `main` branch.

3. **Documentation**:

   - If your pull request introduces new features or changes existing functionality, update the relevant documentation.

4. **Labeling**:

   - Use appropriate labels (e.g., `bug`, `enhancement`, `documentation`) to categorize the pull request.

5. **Merging**:
   - Once approved, passing all checks, and reviewed by the required number of maintainers, the pull request can be merged.
   - Preferably use "Squash and merge" to maintain a clean commit history, unless the commit history is meaningful and should be preserved.

## Branch Management Policy

1. **Branch Naming Conventions**:

   - Use descriptive names for your branches to make it clear what feature or fix is being worked on.
   - Common prefixes include `feature/`, `bugfix/`, `hotfix/`, and `chore/`.
   - Example: `feature/add-login-page`, `bugfix/fix-navbar-issue`.

2. **Creating Branches**:

   - Always create branches from the latest version of the `main` branch.
   - Use short-lived branches for new features, bug fixes, or any changes to the codebase.

3. **Pull Requests**:

   - Ensure your branch is up-to-date with `main` before opening a pull request.
   - Provide a clear and detailed description of the changes in your pull request.
   - Link to any relevant issues or discussions.

4. **Merging Branches**:

   - Ensure all CI checks pass before merging.
   - Preferably use "Squash and merge" to maintain a clean commit history.
   - Obtain necessary approvals from maintainers or reviewers before merging.

5. **Deleting Merged Branches**:

   - After a branch has been successfully merged into `main`, it should be deleted.
   - This helps keep the repository clean and reduces clutter.
   - You can delete the branch directly on GitHub after merging, or use the following Git command:

   ```bash
   git branch -d your-branch-name
   git push origin --delete your-branch-name
   ```
