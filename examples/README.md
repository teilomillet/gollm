# gollm Package Examples

This directory contains example scripts demonstrating various features of the `gollm` package. These examples are designed to help you understand and utilize the capabilities of the package in your projects.

## Prerequisites

- Go 1.16 or later
- API keys for the LLM providers you intend to use (e.g., OpenAI, Anthropic, Groq)

Before running the examples, make sure to set up your API keys as environment variables:

```bash
export OPENAI_API_KEY=your_api_key_here
export ANTHROPIC_API_KEY=your_api_key_here
export GROQ_API_KEY=your_api_key_here
```

Alternatively, you can use a `.env` file in the examples directory:

```bash
OPENAI_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=your_api_key_here
GROQ_API_KEY=your_api_key_here
```

## Running the Examples

To run an example, navigate to this directory and use the `go run` command:

```bash
go run 1_basic_usage.go
```

## List of Examples

1. `1_basic_usage.go`: Demonstrates basic usage of the `gollm` package, including creating an LLM client, generating responses, and using various prompt types.
2. `2_prompt_types.go`: Shows different prompt types and how to use them (basic, with directives, with context, with output, with examples, and prompt templates).
3. `3_compare_providers.go`: Illustrates how to compare responses from different LLM providers and models.
4. `4_custom_config.go`: Demonstrates how to use custom configurations for LLM clients.
5. `5_advanced_prompt.go`: Shows advanced prompt engineering techniques, including chaining different operations.
6. `6_structured_output.go`: Demonstrates how to generate and validate structured output using JSON schemas.
7. `7_structured_output_comparison.go`: Compares structured output generation across different providers.
8. `chain_of_thought_example.go`: Illustrates the use of the Chain of Thought feature for complex reasoning tasks.
9. `question_answer_example.go`: Shows how to use the QuestionAnswer function for simple Q&A tasks.
10. `summarize_example.go`: Demonstrates the use of the summarization feature.
11. `mixture_of_agents_example.go`: Demonstrates how to use the Mixture of Agents feature for improving outputs using multiple agents.

## Configuration

Some examples use the default configuration, while others may require specific config files. To set up custom configurations, create YAML files in the `~/.gollm/configs/` directory. For example:

```yaml
# ~/.gollm/configs/openai.yaml
provider: openai
model: gpt-4o-mini
temperature: 0.7
max_tokens: 100

# ~/.gollm/configs/anthropic.yaml
provider: anthropic
model: claude-3-opus-20240229
temperature: 0.7
max_tokens: 100
```

Adjust these configurations as needed for your use case.

## Troubleshooting

- If you encounter "API key not found" errors, ensure you've correctly set up your environment variables or .env file.
- For "provider not supported" errors, check that you're using a supported LLM provider and that the provider name is correctly specified in your configuration.
- If you're having issues with a specific example, try running the `1_basic_usage.go` example first to ensure your setup is correct.

For more detailed information on using the `gollm` package, refer to the main README.md file in the project root directory. package, refer to the main README.md file in the project root directory.
