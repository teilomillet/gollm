// Package llm provides a unified interface for interacting with various Language Learning Model providers.
//
//nolint:gochecknoglobals // This needs to be refactored, but not today.
package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"sync"

	"github.com/go-playground/validator/v10"
)

const (
	typeObject  = "object"
	typeString  = "string"
	typeInteger = "integer"
	typeNumber  = "number"
	typeBoolean = "boolean"
	typeArray   = "array"
)

// validate is the shared validator instance used across the package.
var validate *validator.Validate
var validateOnce sync.Once

// getValidator returns the initialized validator instance, creating it if necessary.
func getValidator() *validator.Validate {
	validateOnce.Do(func() {
		validate = validator.New()

		// Register custom validator for API key map
		if err := validate.RegisterValidation("apikey", validateAPIKey); err != nil {
			// This is a critical setup failure
			panic(fmt.Sprintf("failed to register API key validator: %v", err))
		}
	})
	return validate
}

// validateAPIKey checks if the API key map contains a valid key for the current provider
func validateAPIKey(fl validator.FieldLevel) bool {
	apiKeys, ok := fl.Field().Interface().(map[string]string)
	if !ok {
		return false
	}

	// Get the parent struct (Config)
	parent := fl.Parent()
	provider := parent.FieldByName("Provider").String()

	// For Ollama, we don't require an API key
	if provider == "ollama" {
		// For Ollama, check if the endpoint is accessible
		endpoint := parent.FieldByName("OllamaEndpoint").String()
		if endpoint == "" {
			endpoint = "http://localhost:11434" // default endpoint
		}
		// Try to make a HEAD request to the Ollama endpoint
		req, err := http.NewRequestWithContext(context.Background(), http.MethodHead, endpoint+"/api/tags", http.NoBody)
		if err != nil {
			return false
		}
		client := &http.Client{Timeout: DefaultOllamaTimeout}
		resp, err := client.Do(req)
		if err != nil {
			return false
		}
		defer func() {
			if closeErr := resp.Body.Close(); closeErr != nil {
				// Log error if needed, but don't fail validation
				_ = closeErr
			}
		}()
		return resp.StatusCode == http.StatusOK
	}

	// For other providers, check if there's a key for the provider
	apiKey, exists := apiKeys[provider]
	if !exists || apiKey == "" {
		return false
	}

	// Validate key format based on provider
	switch provider {
	case "openai":
		return strings.HasPrefix(apiKey, "sk-") && len(apiKey) > MinAPIKeyLength
	case "anthropic":
		return strings.HasPrefix(apiKey, "sk-ant-") && len(apiKey) > MinAPIKeyLength
	default:
		return len(apiKey) > MinAPIKeyLength // Generic validation for unknown providers
	}
}

// Validate checks if the given struct is valid according to its validation rules.
// It uses the go-playground/validator package to perform validation based on struct tags.
//
// Parameters:
//   - s: The struct to validate. Must be a pointer to a struct.
//
// Returns:
//   - error: nil if validation passes, otherwise returns validation errors
//
// Example:
//
//	type Config struct {
//	    Model     string `validate:"required"`
//	    MaxTokens int    `validate:"min=1,max=4096"`
//	}
//
//	config := Config{Model: "gpt-4", MaxTokens: 2048}
//	if err := Validate(&config); err != nil {
//	    log.Fatal(err)
//	}
func Validate(s any) error {
	if err := getValidator().Struct(s); err != nil {
		return fmt.Errorf("validation failed: %w", err)
	}
	return nil
}

// RegisterCustomValidation registers a custom validation function with the validator.
// This allows adding domain-specific validation rules beyond the standard ones.
//
// Parameters:
//   - tag: The validation tag to register (e.g., "customcheck")
//   - fn: The validation function to register
//
// Returns:
//   - error: nil if registration succeeds, otherwise returns an error
//
// Example:
//
//	func validateModel(fl validator.FieldLevel) bool {
//	    return strings.HasPrefix(fl.Field().String(), "gpt-")
//	}
//
//	err := RegisterCustomValidation("model", validateModel)
func RegisterCustomValidation(tag string, fn validator.Func) error {
	if err := getValidator().RegisterValidation(tag, fn); err != nil {
		return fmt.Errorf("failed to register validation for tag '%s': %w", tag, err)
	}
	return nil
}

// GenerateJSONSchema generates a JSON schema for the given struct.
// The schema includes type information, validation rules, and nested structures.
//
// Parameters:
//   - v: The struct to generate schema for
//
// Returns:
//   - []byte: The generated JSON schema
//   - error: Any error encountered during generation
//
// Example:
//
//	type Prompt struct {
//	    Text      string   `json:"text" validate:"required"`
//	    MaxTokens int      `json:"max_tokens" validate:"min=1"`
//	    Stop      []string `json:"stop,omitempty"`
//	}
//
//	schema, err := GenerateJSONSchema(&Prompt{})
func GenerateJSONSchema(v any) ([]byte, error) {
	schema := make(map[string]any)
	schema["type"] = typeObject
	properties, required, err := getStructProperties(reflect.TypeOf(v))
	if err != nil {
		return nil, err
	}
	schema["properties"] = properties
	if len(required) > 0 {
		schema["required"] = required
	}
	data, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("failed to marshal schema: %w", err)
	}
	return data, nil
}

// getStructProperties analyzes a struct type and returns its JSON schema properties.
// It processes struct fields, their types, and validation rules to build the schema.
//
// Parameters:
//   - t: The reflect.Type of the struct to analyze
//
// Returns:
//   - map[string]any: Schema properties
//   - []string: List of required fields
//   - error: Any error encountered during analysis
func getStructProperties(t reflect.Type) (map[string]any, []string, error) {
	properties := make(map[string]any)
	var required []string

	for i := range t.NumField() {
		field := t.Field(i)
		jsonTag := field.Tag.Get("json")
		if jsonTag == "-" {
			continue
		}
		jsonName := strings.Split(jsonTag, ",")[0]
		if jsonName == "" {
			jsonName = field.Name
		}

		fieldSchema, err := getFieldSchema(&field)
		if err != nil {
			return nil, nil, err
		}
		properties[jsonName] = fieldSchema

		if validateTag := field.Tag.Get("validate"); strings.Contains(validateTag, "required") {
			required = append(required, jsonName)
		}
	}

	return properties, required, nil
}

// getFieldSchema generates a JSON schema for a single struct field.
// It handles various Go types and their corresponding JSON schema representations.
//
// Parameters:
//   - field: The reflect.StructField to generate schema for
//
// Returns:
//   - map[string]any: Field schema
//   - error: Any error encountered during generation
func getFieldSchema(field *reflect.StructField) (map[string]any, error) {
	schema := make(map[string]any)

	switch field.Type.Kind() {
	case reflect.String:
		schema["type"] = typeString
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		schema["type"] = typeInteger
	case reflect.Float32, reflect.Float64:
		schema["type"] = typeNumber
	case reflect.Bool:
		schema["type"] = typeBoolean
	case reflect.Slice:
		schema["type"] = typeArray
		itemSchema, err := getFieldSchema(&reflect.StructField{Type: field.Type.Elem()})
		if err != nil {
			return nil, err
		}
		schema["items"] = itemSchema
	case reflect.Struct:
		schema["type"] = typeObject
		properties, required, err := getStructProperties(field.Type)
		if err != nil {
			return nil, err
		}
		schema["properties"] = properties
		if len(required) > 0 {
			schema["required"] = required
		}
	default:
		return nil, fmt.Errorf("unsupported type: %v", field.Type.Kind())
	}

	addValidationToSchema(schema, field.Tag.Get("validate"))

	return schema, nil
}

// addValidationToSchema adds validation rules from struct tags to the JSON schema.
// It converts Go validation rules to their JSON Schema equivalents.
//
// Parameters:
//   - schema: The schema to add validation rules to
//   - validateTag: The validation tag string to process
func addValidationToSchema(schema map[string]any, validateTag string) {
	rules := strings.Split(validateTag, ",")
	for _, rule := range rules {
		parts := strings.SplitN(rule, "=", MaxValidationSplitParts)
		key := parts[0]
		var value string
		if len(parts) > 1 {
			value = parts[1]
		}
		applyValidationRule(schema, key, value)
	}
}

// applyValidationRule applies a single validation rule to the schema
func applyValidationRule(schema map[string]any, key, value string) {
	switch key {
	case "required":
		// This is handled in generateJSONSchemaFromStruct
	case "min":
		applyMinRule(schema, value)
	case "max":
		applyMaxRule(schema, value)
	case "len":
		applyLenRule(schema, value)
	case "one_decimal":
		schema["multipleOf"] = 0.1
	case "email":
		schema["format"] = "email"
	case "url":
		schema["format"] = "uri"
	case "datetime":
		schema["format"] = "date-time"
	case "regex":
		schema["pattern"] = value
	case "enum":
		schema["enum"] = strings.Split(value, "|")
	case "contains":
		applyContainsRule(schema, value)
	case "excludes":
		applyExcludesRule(schema, value)
	case "unique":
		if value == "true" {
			schema["uniqueItems"] = true
		}

	case "minItems":
		if num, err := strconv.ParseInt(value, 10, 64); err == nil {
			schema["minItems"] = num
		}

	case "maxItems":
		if num, err := strconv.ParseInt(value, 10, 64); err == nil {
			schema["maxItems"] = num
		}

	case "password":
		// Example: password=strong (requires at least 8 characters, 1 uppercase, 1 lowercase, 1 number, 1 special
		// char)
		schema["pattern"] = "^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d)(?=.*[@$!%*?&])[A-Za-z\\d@$!%*?&]{8,}$"
		// Add more cases as needed
	}
}

// applyMinRule applies minimum value validation
func applyMinRule(schema map[string]any, value string) {
	if num, err := strconv.ParseFloat(value, 64); err == nil {
		if schema["type"] == "array" {
			schema["minItems"] = int(num)
		} else {
			schema["minimum"] = num
		}
	}
}

// applyMaxRule applies maximum value validation
func applyMaxRule(schema map[string]any, value string) {
	if num, err := strconv.ParseFloat(value, 64); err == nil {
		if schema["type"] == "array" {
			schema["maxItems"] = int(num)
		} else {
			schema["maximum"] = num
		}
	}
}

// applyLenRule applies length validation
func applyLenRule(schema map[string]any, value string) {
	if num, err := strconv.ParseInt(value, 10, 64); err == nil {
		schema["minLength"] = num
		schema["maxLength"] = num
	}
}

// applyContainsRule applies contains validation
func applyContainsRule(schema map[string]any, value string) {
	if schema["allOf"] == nil {
		schema["allOf"] = []map[string]any{}
	}
	allOf, ok := schema["allOf"].([]map[string]any)
	if !ok {
		return
	}
	schema["allOf"] = append(allOf,
		map[string]any{
			"pattern": fmt.Sprintf(".*%s.*", regexp.QuoteMeta(value)),
		})
}

// applyExcludesRule applies excludes validation
func applyExcludesRule(schema map[string]any, value string) {
	if schema["not"] == nil {
		schema["not"] = map[string]any{}
	}
	notSchema, ok := schema["not"].(map[string]any)
	if ok {
		notSchema["pattern"] = fmt.Sprintf(".*%s.*", regexp.QuoteMeta(value))
	}
}

// ValidateAgainstSchema validates a JSON response against a JSON schema.
// It ensures the response matches the expected structure and constraints.
//
// Parameters:
//   - response: The JSON response string to validate
//   - schema: The schema to validate against
//
// Returns:
//   - error: nil if validation passes, otherwise returns validation errors
//
// Example:
//
//	schema := map[string]any{
//	    "type": "object",
//	    "properties": map[string]any{
//	        "text": map[string]any{
//	            "type": "string",
//	        },
//	    },
//	}
//
//	err := ValidateAgainstSchema(`{"text": "Hello"}`, schema)
func ValidateAgainstSchema(response string, schema any) error {
	var responseData any
	if err := json.Unmarshal([]byte(response), &responseData); err != nil {
		return fmt.Errorf("failed to parse response JSON: %w", err)
	}

	var schemaMap map[string]any
	switch s := schema.(type) {
	case string:
		if err := json.Unmarshal([]byte(s), &schemaMap); err != nil {
			return fmt.Errorf("failed to parse schema JSON string: %w", err)
		}
	case []byte:
		if err := json.Unmarshal(s, &schemaMap); err != nil {
			return fmt.Errorf("failed to parse schema JSON bytes: %w", err)
		}
	case map[string]any:
		schemaMap = s
	default:
		// Try to marshal and unmarshal to ensure we have a proper object
		schemaBytes, err := json.Marshal(schema)
		if err != nil {
			return fmt.Errorf("failed to marshal schema: %w", err)
		}
		if err := json.Unmarshal(schemaBytes, &schemaMap); err != nil {
			return fmt.Errorf("failed to parse schema JSON: %w", err)
		}
	}

	if err := validateJSONAgainstSchema(responseData, schemaMap); err != nil {
		return fmt.Errorf("response does not match schema: %w", err)
	}

	return nil
}

// validateJSONAgainstSchema performs the actual JSON schema validation.
// It recursively validates complex data structures against their schema.
//
// Parameters:
//   - data: The data to validate
//   - schema: The schema to validate against
//
// Returns:
//   - error: nil if validation passes, otherwise returns validation errors
func validateJSONAgainstSchema(data any, schema map[string]any) error {
	schemaType, ok := schema["type"].(string)
	if !ok {
		return errors.New("schema missing 'type' field")
	}

	switch schemaType {
	case typeObject:
		return validateObject(data, schema)
	case typeArray:
		return validateArray(data, schema)
	case typeString, typeNumber, typeInteger, typeBoolean:
		return validatePrimitive(data, schemaType)
	default:
		return fmt.Errorf("unsupported schema type: %s", schemaType)
	}
}

// validateObject validates an object against its schema.
// It checks object properties and their types according to the schema.
//
// Parameters:
//   - data: The object to validate
//   - schema: The schema to validate against
//
// Returns:
//   - error: nil if validation passes, otherwise returns validation errors
func validateObject(data any, schema map[string]any) error {
	dataMap, ok := data.(map[string]any)
	if !ok {
		return fmt.Errorf("expected object, got %T", data)
	}

	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		return errors.New("invalid 'properties' in schema")
	}

	for key, propSchema := range properties {
		if err := validateObjectProperty(key, propSchema, dataMap, schema); err != nil {
			return err
		}
	}

	return nil
}

// validateObjectProperty validates a single property of an object
func validateObjectProperty(key string, propSchema any, dataMap map[string]any, schema map[string]any) error {
	propData, exists := dataMap[key]
	if !exists {
		return checkRequiredField(key, schema)
	}

	propSchemaMap, ok := propSchema.(map[string]any)
	if !ok {
		return nil // Skip if not a valid schema map
	}

	if err := validateJSONAgainstSchema(propData, propSchemaMap); err != nil {
		return fmt.Errorf("invalid field '%s': %w", key, err)
	}

	return nil
}

// checkRequiredField checks if a field is required and missing
func checkRequiredField(key string, schema map[string]any) error {
	required, ok := schema["required"].([]any)
	if !ok {
		return nil // No required fields specified
	}

	for _, req := range required {
		reqStr, ok := req.(string)
		if ok && reqStr == key {
			return fmt.Errorf("missing required field: %s", key)
		}
	}

	return nil // Field is not required
}

// validateArray validates an array against its schema.
// It checks array items and their types according to the schema.
//
// Parameters:
//   - data: The array to validate
//   - schema: The schema to validate against
//
// Returns:
//   - error: nil if validation passes, otherwise returns validation errors
func validateArray(data any, schema map[string]any) error {
	dataSlice, ok := data.([]any)
	if !ok {
		return fmt.Errorf("expected array, got %T", data)
	}

	items, ok := schema["items"].(map[string]any)
	if !ok {
		return errors.New("invalid 'items' in schema")
	}

	for i, item := range dataSlice {
		if err := validateJSONAgainstSchema(item, items); err != nil {
			return fmt.Errorf("invalid item at index %d: %w", i, err)
		}
	}

	return nil
}

// validatePrimitive validates a primitive value against its expected type.
// It ensures the value matches the type specified in the schema.
//
// Parameters:
//   - data: The value to validate
//   - expectedType: The expected JSON Schema type
//
// Returns:
//   - error: nil if validation passes, otherwise returns validation errors
func validatePrimitive(data any, expectedType string) error {
	switch expectedType {
	case "string":
		if _, ok := data.(string); !ok {
			return fmt.Errorf("expected string, got %T", data)
		}
	case "number":
		if _, ok := data.(float64); !ok {
			return fmt.Errorf("expected number, got %T", data)
		}
	case "integer":
		if _, ok := data.(float64); !ok {
			return fmt.Errorf("expected integer, got %T", data)
		}
	case "boolean":
		if _, ok := data.(bool); !ok {
			return fmt.Errorf("expected boolean, got %T", data)
		}
	}
	return nil
}
