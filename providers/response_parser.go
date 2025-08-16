package providers

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// ExtractFunctionCalls extracts JSON function calls encapsulated within <function_call> tags.
// It returns a slice of function call objects, each containing a name and arguments.
func ExtractFunctionCalls(response string) ([]map[string]any, error) {
	functionCallRegex := regexp.MustCompile(`<function_call>(.*?)</function_call>`)
	matches := functionCallRegex.FindAllStringSubmatch(response, -1)

	functionCalls := make([]map[string]any, 0, len(matches))
	for _, match := range matches {
		if len(match) <= 1 {
			continue
		}

		var functionCall map[string]any
		if err := json.Unmarshal([]byte(match[1]), &functionCall); err != nil {
			return nil, fmt.Errorf("error unmarshaling function call: %w", err)
		}

		// Handle string arguments by attempting to parse them as JSON
		if args, ok := functionCall["arguments"].(string); ok {
			var parsedArgs map[string]any
			if err := json.Unmarshal([]byte(args), &parsedArgs); err != nil {
				return nil, fmt.Errorf("error parsing string arguments: %w", err)
			}
			functionCall["arguments"] = parsedArgs
		}

		functionCalls = append(functionCalls, functionCall)
	}
	return functionCalls, nil
}

// CleanResponse processes a raw LLM response and extracts both the text content
// and any function calls. It returns the cleaned text and a slice of function call JSON strings.
func CleanResponse(rawResponse string) (string, []string, error) {
	var cleanedResponse strings.Builder
	functionCalls := make([]string, 0, ResponseParserRetryAttempts)

	// Extract function calls
	functionCallRegex := regexp.MustCompile(`<function_call>(.*?)</function_call>`)
	matches := functionCallRegex.FindAllStringSubmatchIndex(rawResponse, -1)

	lastIndex := 0
	for _, match := range matches {
		// Append text before the function call
		cleanedResponse.WriteString(rawResponse[lastIndex:match[0]])

		// Extract the function call JSON
		functionCallJSON := rawResponse[match[2]:match[3]]
		functionCalls = append(functionCalls, functionCallJSON)

		lastIndex = match[1]
	}

	// Append any remaining text after the last function call
	cleanedResponse.WriteString(rawResponse[lastIndex:])

	return cleanedResponse.String(), functionCalls, nil
}

// FormatFunctionCall creates a properly formatted function call string
// that can be embedded in the response.
func FormatFunctionCall(name string, arguments any) (string, error) {
	// If arguments is a string, try to parse it as JSON first
	if argsStr, ok := arguments.(string); ok {
		var parsedArgs map[string]any
		if err := json.Unmarshal([]byte(argsStr), &parsedArgs); err == nil {
			arguments = parsedArgs
		}
	}

	functionCall := map[string]any{
		"name":      name,
		"arguments": arguments,
	}

	callJSON, err := json.Marshal(functionCall)
	if err != nil {
		return "", fmt.Errorf("error marshaling function call: %w", err)
	}

	return fmt.Sprintf("<function_call>%s</function_call>", string(callJSON)), nil
}
