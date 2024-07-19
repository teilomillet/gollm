# goal Package Examples

This directory contains example scripts demonstrating various features of the `goal` package. These examples are designed to help you understand and utilize the capabilities of the package in your projects.

## Prerequisites

- Go 1.16 or later
- API keys for the LLM providers you intend to use (e.g., Anthropic, OpenAI)

Before running the examples, make sure to set up your API keys as environment variables:

```bash
export ANTHROPIC_API_KEY=your_api_key_here
export OPENAI_API_KEY=your_api_key_here
```

Alternatively, you can use a `.env` file in the examples directory:

```
ANTHROPIC_API_KEY=your_api_key_here
OPENAI_API_KEY=your_api_key_here
```

## Running the Examples

To run an example, navigate to this directory and use the `go run` command:

```bash
go run 1_basic_usage.go
```

## List of Examples

1. `1_basic_usage.go`: Demonstrates basic usage of the `goal` package, including creating an LLM client and generating responses.
2. `2_prompt_types.go`: Shows different prompt types and how to use them (basic, with directives, with context, with max length, with examples).
3. `3_question_answer.go`: Illustrates how to use the QuestionAnswer function for simple Q&A tasks.
4. `4_custom_prompt.go`: Demonstrates how to create and use custom prompt templates for more complex tasks.
5. `5_advanced_prompt.go`: Shows advanced prompt engineering techniques, including chaining different operations.

## Configuration

Some examples use the default configuration, while others may require specific config files. To set up custom configurations, create YAML files in the `~/.goal/configs/` directory. For example:

```yaml
# ~/.goal/configs/anthropic.yaml
provider: anthropic
model: claude-3-opus-20240229
temperature: 0.7
max_tokens: 100
log_level: info

# ~/.goal/configs/openai.yaml
provider: openai
model: gpt-4
temperature: 0.7
max_tokens: 100
log_level: info
```

Adjust these configurations as needed for your use case.

## Troubleshooting

- If you encounter "API key not found" errors, ensure you've correctly set up your environment variables or .env file.
- For "provider not supported" errors, check that you're using a supported LLM provider and that the provider name is correctly specified in your configuration.
- If you're having issues with a specific example, try running the `1_basic_usage.go` example first to ensure your setup is correct.

For more detailed information on using the `goal` package, refer to the main README.md file in the project root directory.
