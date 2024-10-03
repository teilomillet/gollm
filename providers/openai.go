package providers

import (
	"encoding/json"
	"fmt"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/utils"
)

// OpenAIProvider implements the Provider interface for OpenAI's API
type OpenAIProvider struct {
	apiKey       string
	model        string
	extraHeaders map[string]string
	options      map[string]interface{}
	logger       utils.Logger
}

// NewOpenAIProvider creates a new OpenAI provider instance
func NewOpenAIProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}
	return &OpenAIProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: extraHeaders,
		options:      make(map[string]interface{}),
		logger:       utils.NewLogger(utils.LogLevelInfo), // Default to INFO level
	}
}

// SetOption sets a specific option for the provider
func (p *OpenAIProvider) SetOption(key string, value interface{}) {
	p.options[key] = value
	p.logger.Debug("Option set", "key", key, "value", value)
}

// SetDefaultOptions sets default options based on the provided configuration
func (p *OpenAIProvider) SetDefaultOptions(config *config.Config) {
	p.SetOption("temperature", config.Temperature)
	p.SetOption("max_tokens", config.MaxTokens)
	if config.Seed != nil {
		p.SetOption("seed", *config.Seed)
	}
	p.logger.Debug("Default options set", "temperature", config.Temperature, "max_tokens", config.MaxTokens, "seed", config.Seed)
}

// Name returns the provider's name
func (p *OpenAIProvider) Name() string {
	return "openai"
}

// Endpoint returns the API endpoint for OpenAI
func (p *OpenAIProvider) Endpoint() string {
	return "https://api.openai.com/v1/chat/completions"
}

// SupportsJSONSchema indicates whether this provider supports JSON schema
func (p *OpenAIProvider) SupportsJSONSchema() bool {
	return true
}

// Headers returns the necessary headers for API requests
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

// PrepareRequest prepares the request body for the API call
func (p *OpenAIProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	p.logger.Debug("Preparing request", "prompt", prompt)
	request := p.createBaseRequest(prompt)
	p.processMessages(request)
	p.addOptions(request, options)

	reqJSON, err := json.Marshal(request)
	if err != nil {
		p.logger.Error("Failed to marshal request", "error", err)
		return nil, err
	}

	p.logger.Debug("Request prepared", "request", string(reqJSON))
	return reqJSON, nil
}

// createBaseRequest creates the base request structure
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

// processMessages processes the messages in the request
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

// processFunctionMessage handles function messages
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

// processToolCalls handles tool calls in messages
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

// addOptions adds options to the request
func (p *OpenAIProvider) addOptions(request map[string]interface{}, options map[string]interface{}) {
	for k, v := range p.options {
		request[k] = v
	}
	for k, v := range options {
		request[k] = v
	}
	p.logger.Debug("Options added to request", "options", options)
}

// PrepareRequestWithSchema prepares a request with a JSON schema
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

// ParseResponse parses the API response
func (p *OpenAIProvider) ParseResponse(body []byte) (string, error) {
	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		// If it's not JSON, return the raw body as a string
		return string(body), nil
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("empty response from API")
	}

	return response.Choices[0].Message.Content, nil
}

// HandleFunctionCalls processes function calls in the response
func (p *OpenAIProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	p.logger.Debug("Handling function calls", "body", string(body))
	var response struct {
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
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
		p.logger.Error("Failed to parse response for function calls", "error", err)
		return nil, fmt.Errorf("error parsing response: %w", err)
	}

	if len(response.Choices) == 0 {
		p.logger.Warn("Empty response from API")
		return nil, fmt.Errorf("empty response from API")
	}

	message := response.Choices[0].Message

	if len(message.ToolCalls) > 0 {
		for i, toolCall := range message.ToolCalls {
			var argsMap map[string]interface{}
			if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &argsMap); err != nil {
				p.logger.Error("Failed to unmarshal function call arguments", "error", err)
				return nil, fmt.Errorf("failed to unmarshal arguments for tool call %d: %w", i, err)
			}
			message.ToolCalls[i].Function.Arguments = string(mustMarshal(argsMap))
			p.logger.Debug("Function call processed", "functionName", toolCall.Function.Name, "arguments", argsMap)
		}
		return json.Marshal(message.ToolCalls)
	}

	p.logger.Debug("No function calls found, returning content", "content", message.Content)
	return []byte(message.Content), nil
}

// mustMarshal is a helper function to marshal JSON and panic on error
func mustMarshal(v interface{}) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}

// SetExtraHeaders sets additional headers for the API request
func (p *OpenAIProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
	p.logger.Debug("Extra headers set", "headers", extraHeaders)
}
