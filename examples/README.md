# goal Package Examples

This directory contains example scripts demonstrating various features of the `goal` package.

## Running the Examples

To run an example, navigate to this directory and use the `go run` command:

```
go run 1_basic_usage.go
```

Make sure you have set up your API keys as environment variables before running the examples:

```
export ANTHROPIC_API_KEY=your_api_key_here
export OPENAI_API_KEY=your_api_key_here
```

## Examples

1. `1_basic_usage.go`: Demonstrates basic usage of the `goal` package.
2. `2_prompt_types.go`: Shows different prompt types (Question-Answer, Chain-of-Thought, Summarize).
3. `3_compare_providers.go`: Compares responses from multiple providers.
4. `4_custom_config.go`: Demonstrates how to use a custom configuration.
5. `5_advanced_prompt.go`: Shows advanced prompt engineering techniques.

## Configuration

Some examples use the default configuration, while others may require specific config files. To set up custom configurations, create YAML files in `~/.goal/configs/` directory. For example:

```yaml
# ~/.goal/configs/anthropic.yaml
provider: anthropic
model: claude-2.0
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
