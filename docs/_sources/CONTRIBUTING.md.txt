# Contributing to SelfiSys

We welcome contributions to the SelfiSys repository! Please follow these guidelines when contributing.

---

## Reporting Issues

If you find a bug or have a suggestion, please open an issue in the [GitHub repository](https://github.com/hoellin/selfisys_public). Include as much detail as possible:
- Steps to reproduce the issue
- Expected vs. actual behaviour
- Relevant error messages or logs (if applicable)
- Suggestions for improvement (if applicable)

If you are unsure whether your issue is a bug or have questions about the code, you are welcome to open an issue for discussion.

## Submitting Contributions

1. Fork the repository and create a new branch for your changes.
2. Ensure your contributions are well-documented and adhere to the highest coding standards.
3. Test your changes thoroughly before submitting:
   - Ensure Jupyter notebooks run without errors from top to bottom.
   - Validate new functionality or fixes using appropriate test cases.
4. Before submitting a pull request, synchronise your fork with the main repository to incorporate upstream changes.
   - Add the main repository as a remote.
        ```bash
        git remote add upstream https://github.com/hoellin/selfisys_public.git
        ```
    - Fetch the latest changes from the upstream repository.
        ```bash
        git fetch upstream
        ```
    - Merge the changes into your local repository.
        ```bash
        git merge upstream/main
        ```
5. Open a pull request describing your changes to the main repository.

## Style Guidelines

Follow best practices for Python coding and Jupyter notebooks.

### Python code

Refer to the [PEP 8 Style Guide](https://pep8.org/) for Python coding standards.

### Jupyter Notebooks

- Use clear and concise Markdown cells to explain code and results.
- Avoid leaving unnecessary output (e.g., debugging print statements).
- Ensure notebooks are runnable from top to bottom.
- For tips on creating clean and effective Jupyter notebooks, see [Jupyter Notebook Best Practices](https://realpython.com/jupyter-notebook-best-practices/).

---

## Questions?

If you have any questions or need further clarification, feel free to [contact the authors](mailto:tristan.hoellinger@iap.fr).