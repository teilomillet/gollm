// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

// BedrockProvider implements the Provider interface for AWS Bedrock.
// It supports various foundation models available through AWS Bedrock,
// including Claude, Llama, Mistral, and others.
//
// Authentication uses AWS credentials from environment variables:
//   - AWS_ACCESS_KEY_ID
//   - AWS_SECRET_ACCESS_KEY
//   - AWS_SESSION_TOKEN (optional, for temporary credentials)
//   - AWS_REGION (defaults to "us-east-1")
type BedrockProvider struct {
	apiKey       string                 // Not used directly; AWS credentials come from env
	model        string                 // Model identifier (e.g., "anthropic.claude-3-sonnet-20240229-v1:0")
	extraHeaders map[string]string      // Additional HTTP headers
	options      map[string]interface{} // Model-specific options
	logger       utils.Logger           // Logger instance
	region       string                 // AWS region
	accessKey    string                 // AWS access key ID
	secretKey    string                 // AWS secret access key
	sessionToken string                 // AWS session token (optional)
}

// NewBedrockProvider creates a new AWS Bedrock provider instance.
//
// Parameters:
//   - apiKey: Not used directly; pass empty string. AWS credentials are read from environment.
//   - model: The model to use (e.g., "anthropic.claude-3-sonnet-20240229-v1:0",
//     "meta.llama3-70b-instruct-v1:0", "mistral.mistral-7b-instruct-v0:2")
//   - extraHeaders: Additional HTTP headers for requests
//
// Environment variables:
//   - AWS_ACCESS_KEY_ID: AWS access key (required)
//   - AWS_SECRET_ACCESS_KEY: AWS secret key (required)
//   - AWS_SESSION_TOKEN: Session token for temporary credentials (optional)
//   - AWS_REGION: AWS region (defaults to "us-east-1")
//
// Returns:
//   - A configured Bedrock Provider instance
func NewBedrockProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}

	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = os.Getenv("AWS_DEFAULT_REGION")
	}
	if region == "" {
		region = "us-east-1"
	}

	return &BedrockProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: extraHeaders,
		options:      make(map[string]interface{}),
		logger:       utils.NewLogger(utils.LogLevelInfo),
		region:       region,
		accessKey:    os.Getenv("AWS_ACCESS_KEY_ID"),
		secretKey:    os.Getenv("AWS_SECRET_ACCESS_KEY"),
		sessionToken: os.Getenv("AWS_SESSION_TOKEN"),
	}
}

// SetLogger configures the logger for the Bedrock provider.
func (p *BedrockProvider) SetLogger(logger utils.Logger) {
	p.logger = logger
}

// Name returns "bedrock" as the provider identifier.
func (p *BedrockProvider) Name() string {
	return "bedrock"
}

// Endpoint returns the AWS Bedrock API endpoint URL for the configured region and model.
func (p *BedrockProvider) Endpoint() string {
	return fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com/model/%s/invoke", p.region, p.model)
}

// SupportsJSONSchema indicates that Bedrock supports JSON schema validation
// for supported models.
func (p *BedrockProvider) SupportsJSONSchema() bool {
	return true
}

// Headers returns the required HTTP headers for Bedrock API requests.
// Note: AWS Signature headers are added dynamically during request preparation.
func (p *BedrockProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type": "application/json",
		"Accept":       "application/json",
	}
	for key, value := range p.extraHeaders {
		headers[key] = value
	}
	return headers
}

// SetOption sets a specific option for the Bedrock provider.
// Supported options include:
//   - temperature: Controls randomness
//   - max_tokens: Maximum tokens in the response
//   - top_p: Nucleus sampling parameter
//   - top_k: Top-k sampling parameter
//   - region: AWS region override
func (p *BedrockProvider) SetOption(key string, value interface{}) {
	if key == "region" {
		if region, ok := value.(string); ok {
			p.region = region
		}
	} else {
		p.options[key] = value
	}
	p.logger.Debug("Option set", "key", key, "value", value)
}

// SetDefaultOptions configures standard options from the global configuration.
func (p *BedrockProvider) SetDefaultOptions(config *config.Config) {
	p.SetOption("temperature", config.Temperature)
	p.SetOption("max_tokens", config.MaxTokens)
	if config.Seed != nil {
		p.SetOption("seed", *config.Seed)
	}
}

// getModelFamily returns the model family for request formatting
func (p *BedrockProvider) getModelFamily() string {
	if strings.HasPrefix(p.model, "anthropic.") {
		return "anthropic"
	}
	if strings.HasPrefix(p.model, "meta.") {
		return "meta"
	}
	if strings.HasPrefix(p.model, "mistral.") {
		return "mistral"
	}
	if strings.HasPrefix(p.model, "amazon.") {
		return "amazon"
	}
	if strings.HasPrefix(p.model, "cohere.") {
		return "cohere"
	}
	if strings.HasPrefix(p.model, "ai21.") {
		return "ai21"
	}
	return "unknown"
}

// PrepareRequest creates the request body for a Bedrock API call.
// The request format varies based on the model family.
func (p *BedrockProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	family := p.getModelFamily()

	switch family {
	case "anthropic":
		return p.prepareAnthropicRequest(prompt, options)
	case "meta":
		return p.prepareMetaRequest(prompt, options)
	case "mistral":
		return p.prepareMistralRequest(prompt, options)
	case "cohere":
		return p.prepareCohereRequest(prompt, options)
	default:
		// Generic format for other models
		return p.prepareGenericRequest(prompt, options)
	}
}

func (p *BedrockProvider) prepareAnthropicRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	maxTokens := 4096
	if mt, ok := p.options["max_tokens"].(int); ok {
		maxTokens = mt
	}
	if mt, ok := options["max_tokens"].(int); ok {
		maxTokens = mt
	}

	request := map[string]interface{}{
		"anthropic_version": "bedrock-2023-05-31",
		"max_tokens":        maxTokens,
		"messages": []map[string]interface{}{
			{"role": "user", "content": prompt},
		},
	}

	// Handle system prompt
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		request["system"] = systemPrompt
	}

	// Add temperature if set
	if temp, ok := p.options["temperature"].(float64); ok {
		request["temperature"] = temp
	}
	if temp, ok := options["temperature"].(float64); ok {
		request["temperature"] = temp
	}

	return json.Marshal(request)
}

func (p *BedrockProvider) prepareMetaRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	maxTokens := 2048
	if mt, ok := p.options["max_tokens"].(int); ok {
		maxTokens = mt
	}
	if mt, ok := options["max_tokens"].(int); ok {
		maxTokens = mt
	}

	// Meta Llama format
	request := map[string]interface{}{
		"prompt":     fmt.Sprintf("[INST] %s [/INST]", prompt),
		"max_gen_len": maxTokens,
	}

	if temp, ok := p.options["temperature"].(float64); ok {
		request["temperature"] = temp
	}
	if temp, ok := options["temperature"].(float64); ok {
		request["temperature"] = temp
	}

	return json.Marshal(request)
}

func (p *BedrockProvider) prepareMistralRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	maxTokens := 4096
	if mt, ok := p.options["max_tokens"].(int); ok {
		maxTokens = mt
	}
	if mt, ok := options["max_tokens"].(int); ok {
		maxTokens = mt
	}

	request := map[string]interface{}{
		"prompt":     fmt.Sprintf("<s>[INST] %s [/INST]", prompt),
		"max_tokens": maxTokens,
	}

	if temp, ok := p.options["temperature"].(float64); ok {
		request["temperature"] = temp
	}
	if temp, ok := options["temperature"].(float64); ok {
		request["temperature"] = temp
	}

	return json.Marshal(request)
}

func (p *BedrockProvider) prepareCohereRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	maxTokens := 4096
	if mt, ok := p.options["max_tokens"].(int); ok {
		maxTokens = mt
	}
	if mt, ok := options["max_tokens"].(int); ok {
		maxTokens = mt
	}

	request := map[string]interface{}{
		"message":    prompt,
		"max_tokens": maxTokens,
	}

	if temp, ok := p.options["temperature"].(float64); ok {
		request["temperature"] = temp
	}
	if temp, ok := options["temperature"].(float64); ok {
		request["temperature"] = temp
	}

	return json.Marshal(request)
}

func (p *BedrockProvider) prepareGenericRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	request := map[string]interface{}{
		"inputText": prompt,
	}

	params := map[string]interface{}{}
	if temp, ok := p.options["temperature"].(float64); ok {
		params["temperature"] = temp
	}
	if mt, ok := p.options["max_tokens"].(int); ok {
		params["maxTokenCount"] = mt
	}
	if len(params) > 0 {
		request["textGenerationConfig"] = params
	}

	return json.Marshal(request)
}

// PrepareRequestWithSchema creates a request that includes JSON schema validation.
func (p *BedrockProvider) PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error) {
	schemaJSON, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("failed to marshal schema: %w", err)
	}

	enhancedPrompt := fmt.Sprintf("%s\n\nPlease respond with a JSON object matching this schema:\n%s", prompt, string(schemaJSON))
	return p.PrepareRequest(enhancedPrompt, options)
}

// ParseResponse extracts the generated text from the Bedrock API response.
func (p *BedrockProvider) ParseResponse(body []byte) (string, error) {
	family := p.getModelFamily()

	switch family {
	case "anthropic":
		return p.parseAnthropicResponse(body)
	case "meta":
		return p.parseMetaResponse(body)
	case "mistral":
		return p.parseMistralResponse(body)
	case "cohere":
		return p.parseCohereResponse(body)
	default:
		return p.parseGenericResponse(body)
	}
}

func (p *BedrockProvider) parseAnthropicResponse(body []byte) (string, error) {
	var response struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("error parsing Anthropic response: %w", err)
	}
	if len(response.Content) == 0 {
		return "", fmt.Errorf("empty response from API")
	}
	var result strings.Builder
	for _, c := range response.Content {
		if c.Type == "text" {
			result.WriteString(c.Text)
		}
	}
	return result.String(), nil
}

func (p *BedrockProvider) parseMetaResponse(body []byte) (string, error) {
	var response struct {
		Generation string `json:"generation"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("error parsing Meta response: %w", err)
	}
	return response.Generation, nil
}

func (p *BedrockProvider) parseMistralResponse(body []byte) (string, error) {
	var response struct {
		Outputs []struct {
			Text string `json:"text"`
		} `json:"outputs"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("error parsing Mistral response: %w", err)
	}
	if len(response.Outputs) == 0 {
		return "", fmt.Errorf("empty response from API")
	}
	return response.Outputs[0].Text, nil
}

func (p *BedrockProvider) parseCohereResponse(body []byte) (string, error) {
	var response struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("error parsing Cohere response: %w", err)
	}
	return response.Text, nil
}

func (p *BedrockProvider) parseGenericResponse(body []byte) (string, error) {
	var response struct {
		Results []struct {
			OutputText string `json:"outputText"`
		} `json:"results"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("error parsing response: %w", err)
	}
	if len(response.Results) == 0 {
		return "", fmt.Errorf("empty response from API")
	}
	return response.Results[0].OutputText, nil
}

// HandleFunctionCalls processes function calling capabilities.
func (p *BedrockProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	response := string(body)
	functionCalls, err := utils.ExtractFunctionCalls(response)
	if err != nil {
		return nil, fmt.Errorf("error extracting function calls: %w", err)
	}
	if len(functionCalls) == 0 {
		return nil, nil
	}
	return json.Marshal(functionCalls)
}

// SetExtraHeaders configures additional HTTP headers for API requests.
func (p *BedrockProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
}

// SupportsStreaming indicates whether streaming is supported.
func (p *BedrockProvider) SupportsStreaming() bool {
	return true
}

// PrepareStreamRequest creates a request body for streaming API calls.
func (p *BedrockProvider) PrepareStreamRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	return p.PrepareRequest(prompt, options)
}

// ParseStreamResponse processes a single chunk from a streaming response.
func (p *BedrockProvider) ParseStreamResponse(chunk []byte) (string, error) {
	if len(bytes.TrimSpace(chunk)) == 0 {
		return "", fmt.Errorf("empty chunk")
	}

	family := p.getModelFamily()
	switch family {
	case "anthropic":
		var event struct {
			Type  string `json:"type"`
			Delta struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"delta"`
		}
		if err := json.Unmarshal(chunk, &event); err != nil {
			return "", err
		}
		if event.Type == "content_block_delta" && event.Delta.Type == "text_delta" {
			return event.Delta.Text, nil
		}
		if event.Type == "message_stop" {
			return "", io.EOF
		}
		return "", fmt.Errorf("skip token")
	default:
		// Generic streaming response
		var response struct {
			Token struct {
				Text string `json:"text"`
			} `json:"token"`
		}
		if err := json.Unmarshal(chunk, &response); err != nil {
			return "", err
		}
		return response.Token.Text, nil
	}
}

// PrepareRequestWithMessages creates a request body using structured message objects.
func (p *BedrockProvider) PrepareRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error) {
	family := p.getModelFamily()

	if family == "anthropic" {
		return p.prepareAnthropicMessagesRequest(messages, options)
	}

	// For other model families, convert messages to a single prompt
	var promptBuilder strings.Builder
	for _, msg := range messages {
		promptBuilder.WriteString(fmt.Sprintf("%s: %s\n", msg.Role, msg.Content))
	}
	return p.PrepareRequest(promptBuilder.String(), options)
}

func (p *BedrockProvider) prepareAnthropicMessagesRequest(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error) {
	maxTokens := 4096
	if mt, ok := p.options["max_tokens"].(int); ok {
		maxTokens = mt
	}
	if mt, ok := options["max_tokens"].(int); ok {
		maxTokens = mt
	}

	request := map[string]interface{}{
		"anthropic_version": "bedrock-2023-05-31",
		"max_tokens":        maxTokens,
		"messages":          []map[string]interface{}{},
	}

	// Handle system prompt
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		request["system"] = systemPrompt
	}

	// Convert messages
	for _, msg := range messages {
		request["messages"] = append(request["messages"].([]map[string]interface{}), map[string]interface{}{
			"role":    msg.Role,
			"content": msg.Content,
		})
	}

	// Add temperature if set
	if temp, ok := p.options["temperature"].(float64); ok {
		request["temperature"] = temp
	}
	if temp, ok := options["temperature"].(float64); ok {
		request["temperature"] = temp
	}

	return json.Marshal(request)
}

// SignRequest adds AWS Signature Version 4 headers to the request.
// This method should be called before making the HTTP request.
func (p *BedrockProvider) SignRequest(req *http.Request, body []byte) error {
	if p.accessKey == "" || p.secretKey == "" {
		return fmt.Errorf("AWS credentials not configured: set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
	}

	now := time.Now().UTC()
	amzDate := now.Format("20060102T150405Z")
	dateStamp := now.Format("20060102")

	// Add required headers
	req.Header.Set("X-Amz-Date", amzDate)
	req.Header.Set("Host", req.Host)
	if p.sessionToken != "" {
		req.Header.Set("X-Amz-Security-Token", p.sessionToken)
	}

	// Calculate payload hash
	payloadHash := sha256Hex(body)
	req.Header.Set("X-Amz-Content-Sha256", payloadHash)

	// Create canonical request
	canonicalURI := req.URL.Path
	canonicalQueryString := req.URL.RawQuery

	// Get sorted header names
	signedHeaders := []string{"content-type", "host", "x-amz-content-sha256", "x-amz-date"}
	if p.sessionToken != "" {
		signedHeaders = append(signedHeaders, "x-amz-security-token")
	}
	sort.Strings(signedHeaders)

	// Create canonical headers
	var canonicalHeaders strings.Builder
	for _, h := range signedHeaders {
		canonicalHeaders.WriteString(h)
		canonicalHeaders.WriteString(":")
		canonicalHeaders.WriteString(strings.TrimSpace(req.Header.Get(h)))
		canonicalHeaders.WriteString("\n")
	}

	signedHeadersStr := strings.Join(signedHeaders, ";")

	canonicalRequest := strings.Join([]string{
		req.Method,
		canonicalURI,
		canonicalQueryString,
		canonicalHeaders.String(),
		signedHeadersStr,
		payloadHash,
	}, "\n")

	// Create string to sign
	algorithm := "AWS4-HMAC-SHA256"
	credentialScope := fmt.Sprintf("%s/%s/bedrock/aws4_request", dateStamp, p.region)
	stringToSign := strings.Join([]string{
		algorithm,
		amzDate,
		credentialScope,
		sha256Hex([]byte(canonicalRequest)),
	}, "\n")

	// Calculate signature
	signingKey := getSignatureKey(p.secretKey, dateStamp, p.region, "bedrock")
	signature := hex.EncodeToString(hmacSHA256(signingKey, []byte(stringToSign)))

	// Create authorization header
	authHeader := fmt.Sprintf("%s Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		algorithm, p.accessKey, credentialScope, signedHeadersStr, signature)

	req.Header.Set("Authorization", authHeader)

	return nil
}

// Helper functions for AWS SigV4 signing

func sha256Hex(data []byte) string {
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}

func hmacSHA256(key, data []byte) []byte {
	h := hmac.New(sha256.New, key)
	h.Write(data)
	return h.Sum(nil)
}

func getSignatureKey(secretKey, dateStamp, region, service string) []byte {
	kDate := hmacSHA256([]byte("AWS4"+secretKey), []byte(dateStamp))
	kRegion := hmacSHA256(kDate, []byte(region))
	kService := hmacSHA256(kRegion, []byte(service))
	kSigning := hmacSHA256(kService, []byte("aws4_request"))
	return kSigning
}
