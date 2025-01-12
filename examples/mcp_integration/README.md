# MCP (Model Context Protocol) Integration Examples

This directory contains examples demonstrating how to use MCP-formatted tools with different LLM providers in gollm.

## What is MCP?

MCP (Model Context Protocol) is a standardized format for defining tools that can be used by language models. In gollm, it provides a consistent way to define tools that work across different providers (OpenAI, Anthropic, etc.).

## Examples Overview

1. **Basic Tool Usage** (`Example1_BasicToolUsage`):
   - Shows how to use a single MCP-formatted tool
   - Demonstrates basic weather tool integration with OpenAI

2. **Multiple Tools** (`Example2_MultipleTools`):
   - Shows how to use multiple tools together
   - Demonstrates weather and time tools with Anthropic

3. **Tool Choice Modes** (`Example3_ToolChoiceModes`):
   - Shows different ways to control tool usage
   - Demonstrates forced vs automatic tool choice

## Running the Examples

1. Set up your API keys:
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   ```

2. Run the examples:
   ```bash
   go run main.go
   ```

## Tool Definitions

The examples use two predefined tools:

1. **Weather Tool**:
   - Gets weather information for a location
   - Requires location parameter
   - Optional unit parameter (celsius/fahrenheit)

2. **Time Tool**:
   - Gets current time for a location
   - Requires location parameter
   - Optional timezone parameter

## Key Concepts

1. **Tool Format**:
   ```json
   {
     "type": "function",
     "name": "tool_name",
     "description": "what the tool does",
     "parameters": {
       "type": "object",
       "properties": {
         // tool parameters
       }
     }
   }
   ```

2. **Tool Choice Modes**:
   - `"auto"`: Let the model decide when to use tools
   - `"any"`: Force the model to use a tool

3. **Multiple Tools**:
   - Tools can be combined in a single prompt
   - Models can choose which tool(s) to use based on the query

## Error Handling

The examples demonstrate proper error handling:
- API key validation
- Tool conversion errors
- Response generation errors

## Testing

The `main_test.go` file contains comprehensive tests for:
- Tool format conversion
- Provider integration
- Different usage scenarios

Run the tests with:
```bash
go test -v
```

For quick tests (skipping integration tests):
```bash
go test -v -short
``` 