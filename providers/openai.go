// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"encoding/json"
	"fmt"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/utils"
)

// OpenAIProvider implements the Provider interface for OpenAI's API.
// It supports GPT models and provides access to OpenAI's language model capabilities,
// including function calling, JSON mode, and structured output validation.
type OpenAIProvider struct {
	apiKey       string                 // API key for authentication
	model        string                 // Model identifier (e.g., "gpt-4", "gpt-4o-mini")
	extraHeaders map[string]string      // Additional HTTP headers
	options      map[string]interface{} // Model-specific options
	logger       utils.Logger           // Logger instance
}

// NewOpenAIProvider creates a new OpenAI provider instance.
// It initializes the provider with the given API key, model, and optional headers.
//
// Parameters:
//   - apiKey: OpenAI API key for authentication
//   - model: The model to use (e.g., "gpt-4", "gpt-3.5-turbo")
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured OpenAI Provider instance
func NewOpenAIProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}
	return &OpenAIProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: extraHeaders,
		options:      make(map[string]interface{}),
		logger:       utils.NewLogger(utils.LogLevelInfo),
	}
}

// SetLogger configures the logger for the OpenAI provider.
// This is used for debugging and monitoring API interactions.
func (p *OpenAIProvider) SetLogger(logger utils.Logger) {
	p.logger = logger
}

// SetOption sets a specific option for the OpenAI provider.
// Supported options include:
//   - temperature: Controls randomness (0.0 to 2.0)
//   - max_tokens: Maximum tokens in the response
//   - top_p: Nucleus sampling parameter
//   - frequency_penalty: Repetition reduction
//   - presence_penalty: Topic steering
//   - seed: Deterministic sampling seed
func (p *OpenAIProvider) SetOption(key string, value interface{}) {
	p.options[key] = value
	p.logger.Debug("Option set", "key", key, "value", value)
}

// SetDefaultOptions configures standard options from the global configuration.
// This includes temperature, max tokens, and sampling parameters.
func (p *OpenAIProvider) SetDefaultOptions(config *config.Config) {
	p.SetOption("temperature", config.Temperature)
	p.SetOption("max_tokens", config.MaxTokens)
	if config.Seed != nil {
		p.SetOption("seed", *config.Seed)
	}
	p.logger.Debug("Default options set", "temperature", config.Temperature, "max_tokens", config.MaxTokens, "seed", config.Seed)
}

// Name returns "openai" as the provider identifier.
func (p *OpenAIProvider) Name() string {
	return "openai"
}

// Endpoint returns the OpenAI API endpoint URL.
// For API version 1, this is "https://api.openai.com/v1/chat/completions".
func (p *OpenAIProvider) Endpoint() string {
	return "https://api.openai.com/v1/chat/completions"
}

// SupportsJSONSchema indicates that OpenAI supports native JSON schema validation
// through its function calling and JSON mode capabilities.
func (p *OpenAIProvider) SupportsJSONSchema() bool {
	return true
}

// Headers returns the required HTTP headers for OpenAI API requests.
// This includes:
//   - Authorization: Bearer token using the API key
//   - Content-Type: application/json
//   - Any additional headers specified via SetExtraHeaders
func (p *OpenAIProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type":  "application/json",
		"Authorization": "Bearer " + p.apiKey,
	}

	for key, value := range p.extraHeaders {
		headers[key] = value
	}

	p.logger.Debug("Headers prepared", "headers", headers)
	return headers
}

// PrepareRequest creates the request body for an OpenAI API call.
// It handles:
//   - Message formatting
//   - System messages
//   - Function/tool definitions
//   - Model-specific options
//
// Parameters:
//   - prompt: The input text or conversation
//   - options: Additional parameters for the request
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *OpenAIProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	request := map[string]interface{}{
		"model": p.model,
		"messages": []map[string]interface{}{
			{
				"role":    "user",
				"content": prompt,
			},
		},
	}

	// Handle tool_choice
	if toolChoice, ok := options["tool_choice"].(string); ok {
		request["tool_choice"] = toolChoice
	}

	// Handle tools
	if tools, ok := options["tools"].([]utils.Tool); ok && len(tools) > 0 {
		openAITools := make([]map[string]interface{}, len(tools))
		for i, tool := range tools {
			openAITools[i] = map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name":        tool.Function.Name,
					"description": tool.Function.Description,
					"parameters":  tool.Function.Parameters,
				},
				"strict": true, // Add this if you want strict mode
			}
		}
		request["tools"] = openAITools
	}

	// Add other options
	for k, v := range p.options {
		if k != "tools" && k != "tool_choice" {
			request[k] = v
		}
	}
	for k, v := range options {
		if k != "tools" && k != "tool_choice" {
			request[k] = v
		}
	}

	return json.Marshal(request)
}

// createBaseRequest initializes the basic request structure.
// This includes the model selection and basic message format.
func (p *OpenAIProvider) createBaseRequest(prompt string) map[string]interface{} {
	var request map[string]interface{}
	if err := json.Unmarshal([]byte(prompt), &request); err != nil {
		p.logger.Debug("Prompt is not a valid JSON, creating standard request", "error", err)
		request = map[string]interface{}{
			"model": p.model,
			"messages": []interface{}{
				map[string]interface{}{
					"role":    "user",
					"content": prompt,
				},
			},
		}
	}
	return request
}

// processMessages handles message formatting and structure.
// It processes system messages, user inputs, and assistant responses.
func (p *OpenAIProvider) processMessages(request map[string]interface{}) {
	p.logger.Debug("Processing messages")
	if messages, ok := request["messages"]; ok {
		switch msg := messages.(type) {
		case []interface{}:
			for i, m := range msg {
				if msgMap, ok := m.(map[string]interface{}); ok {
					p.processFunctionMessage(msgMap)
					p.processToolCalls(msgMap)
					msg[i] = msgMap
				}
			}
		case []map[string]string:
			newMessages := make([]interface{}, len(msg))
			for i, m := range msg {
				msgMap := make(map[string]interface{})
				for k, v := range m {
					msgMap[k] = v
				}
				p.processFunctionMessage(msgMap)
				p.processToolCalls(msgMap)
				newMessages[i] = msgMap
			}
			request["messages"] = newMessages
		default:
			p.logger.Warn("Unexpected type for messages", "type", fmt.Sprintf("%T", messages))
		}
	}
	p.logger.Debug("Messages processed", "messageCount", len(request["messages"].([]interface{})))
}

// processFunctionMessage handles function call responses.
// This is used when the model has executed a function and needs to process the result.
func (p *OpenAIProvider) processFunctionMessage(msgMap map[string]interface{}) {
	if msgMap["role"] == "function" && msgMap["name"] == nil {
		if content, ok := msgMap["content"].(string); ok {
			var contentMap map[string]interface{}
			if err := json.Unmarshal([]byte(content), &contentMap); err == nil {
				if name, ok := contentMap["name"].(string); ok {
					msgMap["name"] = name
					p.logger.Debug("Function name extracted from content", "name", name)
				}
			}
		}
	}
}

// processToolCalls handles tool/function definitions and responses.
// This supports OpenAI's function calling capability.
func (p *OpenAIProvider) processToolCalls(msgMap map[string]interface{}) {
	if toolCalls, ok := msgMap["tool_calls"].([]interface{}); ok {
		for j, call := range toolCalls {
			if callMap, ok := call.(map[string]interface{}); ok {
				if function, ok := callMap["function"].(map[string]interface{}); ok {
					if args, ok := function["arguments"].(string); ok {
						var parsedArgs map[string]interface{}
						if err := json.Unmarshal([]byte(args), &parsedArgs); err == nil {
							function["arguments"] = parsedArgs
							callMap["function"] = function
							toolCalls[j] = callMap
							p.logger.Debug("Tool call arguments parsed", "functionName", function["name"], "arguments", parsedArgs)
						}
					}
				}
			}
		}
		msgMap["tool_calls"] = toolCalls
	}
}

// addOptions incorporates additional options into the request.
// This includes model parameters and any provider-specific settings.
func (p *OpenAIProvider) addOptions(request map[string]interface{}, options map[string]interface{}) {
	for k, v := range p.options {
		request[k] = v
	}
	for k, v := range options {
		request[k] = v
	}
	p.logger.Debug("Options added to request", "options", options)
}

// PrepareRequestWithSchema creates a request that includes JSON schema validation.
// This uses OpenAI's function calling feature to enforce response structure.
//
// Parameters:
//   - prompt: The input text or conversation
//   - options: Additional request parameters
//   - schema: JSON schema for response validation
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *OpenAIProvider) PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error) {
	p.logger.Debug("Preparing request with schema", "prompt", prompt, "schema", schema)
	request := map[string]interface{}{
		"model": p.model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"response_format": map[string]interface{}{
			"type":   "json_schema",
			"schema": schema,
		},
	}

	for k, v := range options {
		request[k] = v
	}

	reqJSON, err := json.Marshal(request)
	if err != nil {
		p.logger.Error("Failed to marshal request with schema", "error", err)
		return nil, err
	}

	p.logger.Debug("Request with schema prepared", "request", string(reqJSON))
	return reqJSON, nil
}

// ParseResponse extracts the generated text from the OpenAI API response.
// It handles various response formats and error cases.
//
// Parameters:
//   - body: Raw API response body
//
// Returns:
//   - Generated text content
//   - Any error encountered during parsing
func (p *OpenAIProvider) ParseResponse(body []byte) (string, error) {
	var response struct {
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
				ToolCalls []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return "", err
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("empty response from API")
	}

	message := response.Choices[0].Message
	if message.Content != "" {
		return message.Content, nil
	}

	if len(message.ToolCalls) > 0 {
		toolCallJSON, err := json.Marshal(message.ToolCalls)
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("<function_call>%s</function_call>", toolCallJSON), nil
	}

	return "", fmt.Errorf("no content or tool calls in response")
}

// HandleFunctionCalls processes function calling in the response.
// This supports OpenAI's function calling and JSON mode features.
func (p *OpenAIProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	var response struct {
		Choices []struct {
			Message struct {
				ToolCalls []struct {
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("error parsing response: %w", err)
	}

	if len(response.Choices) == 0 || len(response.Choices[0].Message.ToolCalls) == 0 {
		return nil, fmt.Errorf("no tool calls found in response")
	}

	toolCalls := response.Choices[0].Message.ToolCalls
	result := make([]map[string]interface{}, len(toolCalls))
	for i, call := range toolCalls {
		var args map[string]interface{}
		if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
			return nil, fmt.Errorf("error parsing arguments: %w", err)
		}
		result[i] = map[string]interface{}{
			"name":      call.Function.Name,
			"arguments": args,
		}
	}

	return json.Marshal(result)
}

// mustMarshal is a helper that panics on JSON marshaling errors.
// This is used internally where marshaling errors indicate a programming error.
func mustMarshal(v interface{}) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}

// SetExtraHeaders configures additional HTTP headers for API requests.
// This allows for custom headers needed for specific features or requirements.
func (p *OpenAIProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
	p.logger.Debug("Extra headers set", "headers", extraHeaders)
}
