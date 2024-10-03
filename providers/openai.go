package providers

import (
	"encoding/json"
	"fmt"
)

type OpenAIProvider struct {
	apiKey       string
	model        string
	extraHeaders map[string]string
}

func NewOpenAIProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}
	return &OpenAIProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: extraHeaders,
	}
}

func (p *OpenAIProvider) Name() string {
	return "openai"
}

func (p *OpenAIProvider) Endpoint() string {
	return "https://api.openai.com/v1/chat/completions"
}

func (p *OpenAIProvider) SupportsJSONSchema() bool {
	return true
}

func (p *OpenAIProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type":  "application/json",
		"Authorization": "Bearer " + p.apiKey,
	}

	for key, value := range p.extraHeaders {
		headers[key] = value
	}

	return headers
}

func (p *OpenAIProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	var request map[string]interface{}
	// Try to unmarshal the prompt as JSON
	if err := json.Unmarshal([]byte(prompt), &request); err != nil {
		// If it's not valid JSON, treat it as a regular prompt
		request = map[string]interface{}{
			"model": p.model,
			"messages": []map[string]string{
				{"role": "user", "content": prompt},
			},
		}
	}
	if messages, ok := request["messages"].([]interface{}); ok {
		for i, msg := range messages {
			if msgMap, ok := msg.(map[string]interface{}); ok {
				if msgMap["role"] == "function" && msgMap["name"] == nil {
					// If it's a function message without a name, try to get the name from the content
					if content, ok := msgMap["content"].(string); ok {
						var contentMap map[string]interface{}
						if err := json.Unmarshal([]byte(content), &contentMap); err == nil {
							if name, ok := contentMap["name"].(string); ok {
								msgMap["name"] = name
								messages[i] = msgMap
							}
						}
					}
				}
				// Handle tool_calls if present
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
									}
								}
							}
						}
					}
					msgMap["tool_calls"] = toolCalls
					messages[i] = msgMap
				}
			}
		}
		request["messages"] = messages
	}
	// Add any additional options
	for k, v := range options {
		request[k] = v
	}
	return json.Marshal(request)
}

func (p *OpenAIProvider) PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error) {
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

	// Add any additional options
	for k, v := range options {
		request[k] = v
	}

	return json.Marshal(request)
}

func (p *OpenAIProvider) ParseResponse(body []byte) (string, error) {
	// fmt.Printf("DEBUG: Raw response in OpenAIProvider ParseResponse: %s\n", string(body))
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

	// Unmarshal the main response
	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("error parsing response: %w", err)
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("empty response from API")
	}

	// Process each tool call
	for _, toolCall := range response.Choices[0].Message.ToolCalls {
		// fmt.Printf("DEBUG: Function name: %s\n", toolCall.Function.Name)
		// fmt.Printf("DEBUG: Raw arguments string: %s\n", toolCall.Function.Arguments)

		var argsMap map[string]interface{}

		// Unmarshal the arguments string into a map
		if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &argsMap); err != nil {
			return "", fmt.Errorf("failed to unmarshal arguments: %w", err)
		}

		// Log the successfully unmarshalled arguments
		// fmt.Printf("DEBUG: Unmarshalled function arguments: %+v\n", argsMap)
	}

	// Return the entire response as a JSON string
	return string(body), nil
}

func (p *OpenAIProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
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

	err := json.Unmarshal(body, &response)
	if err != nil {
		return nil, fmt.Errorf("error parsing response: %w", err)
	}

	if len(response.Choices) == 0 {
		return nil, fmt.Errorf("empty response from API")
	}

	message := response.Choices[0].Message

	if len(message.ToolCalls) > 0 {
		// Parse the arguments for each tool call
		for i, toolCall := range message.ToolCalls {
			var argsMap map[string]interface{}
			if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &argsMap); err != nil {
				return nil, fmt.Errorf("failed to unmarshal arguments for tool call %d: %w", i, err)
			}
			// Replace the string arguments with the parsed map
			message.ToolCalls[i].Function.Arguments = string(mustMarshal(argsMap))
		}
		return json.Marshal(message.ToolCalls)
	}

	return []byte(message.Content), nil
}

// Helper function to marshal JSON and panic on error
func mustMarshal(v interface{}) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}

func (p *OpenAIProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
}
