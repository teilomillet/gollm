# Contributing to goal

We're excited that you're interested in contributing to the `goal` package! This document outlines the process for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```
   git clone https://github.com/your-username/goal.git
   ```
3. Create a branch for your changes:
   ```
   git checkout -b your-branch-name
   ```

## Making Changes

1. Make your changes in your branch.
2. Add or update tests as necessary.
3. Ensure all tests pass:
   ```
   go test ./...
   ```
4. Update documentation if you're changing functionality.

## Submitting Changes

1. Push your changes to your fork on GitHub.
2. Submit a pull request to the main repository.
3. In your pull request description, explain your changes and the reason for them.

## Pull Request Guidelines

- Keep pull requests focused on a single change or feature.
- Follow Go best practices and style guides.
- Ensure your code is properly formatted (use `gofmt`).
- Write clear, concise commit messages.
- Include tests for new features or bug fixes.
- Update documentation as necessary.

## Reporting Bugs

- Use the GitHub issue tracker to report bugs.
- Describe the bug in detail, including steps to reproduce.
- Include the version of `goal` you're using and your Go version.

## Suggesting Enhancements

- Use the GitHub issue tracker to suggest enhancements.
- Clearly describe the enhancement and its potential benefits.
- Understand that the project maintainers have the final say on whether to implement suggested enhancements.

## Adding New Providers

If you want to add support for a new LLM provider:

1. Create a new file named after the provider (e.g., `newprovider.go`).
2. Implement the `Provider` interface for the new provider.
3. Add the new provider to the provider registry in `provider_registry.go`.
4. Write tests for the new provider.
5. Update documentation to include the new provider.

## Code Style

- Follow the [Go Code Review Comments](https://github.com/golang/go/wiki/CodeReviewComments) for style guidance.
- Use `gofmt` to format your code before submitting.
- Write idiomatic Go code.

## License

By contributing to `goal`, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

Thank you for contributing to `goal`!
