// Package llm provides a unified interface for interacting with various Language Learning Model providers.
package llm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"regexp"
	"strconv"
	"strings"

	"github.com/go-playground/validator/v10"
)

// validate is the shared validator instance used across the package.
var validate *validator.Validate

func init() {
	validate = validator.New()

	// Register custom validator for API key map
	if err := validate.RegisterValidation("apikey", validateAPIKey); err != nil {
		// Since this is in init(), we can't return the error
		// Instead, panic with a clear message as this is a critical setup failure
		panic(fmt.Sprintf("failed to register API key validator: %v", err))
	}
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
		resp, err := http.Head(endpoint + "/api/tags")
		if err != nil {
			return false
		}
		defer resp.Body.Close()
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
		return strings.HasPrefix(apiKey, "sk-") && len(apiKey) > 20
	case "anthropic":
		return strings.HasPrefix(apiKey, "sk-ant-") && len(apiKey) > 20
	default:
		return len(apiKey) > 20 // Generic validation for unknown providers
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
func Validate(s interface{}) error {
	return validate.Struct(s)
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
	return validate.RegisterValidation(tag, fn)
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
func GenerateJSONSchema(v interface{}) ([]byte, error) {
	schema := make(map[string]interface{})
	schema["type"] = "object"
	properties, required, err := getStructProperties(reflect.TypeOf(v))
	if err != nil {
		return nil, err
	}
	schema["properties"] = properties
	if len(required) > 0 {
		schema["required"] = required
	}
	return json.MarshalIndent(schema, "", "  ")
}

// getStructProperties analyzes a struct type and returns its JSON schema properties.
// It processes struct fields, their types, and validation rules to build the schema.
//
// Parameters:
//   - t: The reflect.Type of the struct to analyze
//
// Returns:
//   - map[string]interface{}: Schema properties
//   - []string: List of required fields
//   - error: Any error encountered during analysis
func getStructProperties(t reflect.Type) (map[string]interface{}, []string, error) {
	properties := make(map[string]interface{})
	var required []string

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		jsonTag := field.Tag.Get("json")
		if jsonTag == "-" {
			continue
		}
		jsonName := strings.Split(jsonTag, ",")[0]
		if jsonName == "" {
			jsonName = field.Name
		}

		fieldSchema, err := getFieldSchema(field)
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
//   - map[string]interface{}: Field schema
//   - error: Any error encountered during generation
func getFieldSchema(field reflect.StructField) (map[string]interface{}, error) {
	schema := make(map[string]interface{})

	switch field.Type.Kind() {
	case reflect.String:
		schema["type"] = "string"
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		schema["type"] = "integer"
	case reflect.Float32, reflect.Float64:
		schema["type"] = "number"
	case reflect.Bool:
		schema["type"] = "boolean"
	case reflect.Slice:
		schema["type"] = "array"
		itemSchema, err := getFieldSchema(reflect.StructField{Type: field.Type.Elem()})
		if err != nil {
			return nil, err
		}
		schema["items"] = itemSchema
	case reflect.Struct:
		schema["type"] = "object"
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
func addValidationToSchema(schema map[string]interface{}, validateTag string) {
	rules := strings.Split(validateTag, ",")
	for _, rule := range rules {
		parts := strings.SplitN(rule, "=", 2)
		key := parts[0]
		var value string
		if len(parts) > 1 {
			value = parts[1]
		}

		switch key {
		case "required":
			// This is handled in generateJSONSchemaFromStruct

		case "min":
			if num, err := strconv.ParseFloat(value, 64); err == nil {
				if schema["type"] == "array" {
					schema["minItems"] = int(num)
				} else {
					schema["minimum"] = num
				}
			}

		case "max":
			if num, err := strconv.ParseFloat(value, 64); err == nil {
				if schema["type"] == "array" {
					schema["maxItems"] = int(num)
				} else {
					schema["maximum"] = num
				}
			}

		case "len":
			if num, err := strconv.ParseInt(value, 10, 64); err == nil {
				schema["minLength"] = num
				schema["maxLength"] = num
			}

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
			if schema["allOf"] == nil {
				schema["allOf"] = []map[string]interface{}{}
			}
			schema["allOf"] = append(schema["allOf"].([]map[string]interface{}),
				map[string]interface{}{
					"pattern": fmt.Sprintf(".*%s.*", regexp.QuoteMeta(value)),
				})

		case "excludes":
			if schema["not"] == nil {
				schema["not"] = map[string]interface{}{}
			}
			schema["not"].(map[string]interface{})["pattern"] = fmt.Sprintf(".*%s.*", regexp.QuoteMeta(value))

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
			// Example: password=strong (requires at least 8 characters, 1 uppercase, 1 lowercase, 1 number, 1 special char)
			schema["pattern"] = "^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d)(?=.*[@$!%*?&])[A-Za-z\\d@$!%*?&]{8,}$"

			// Add more cases as needed
		}
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
//	schema := map[string]interface{}{
//	    "type": "object",
//	    "properties": map[string]interface{}{
//	        "text": map[string]interface{}{
//	            "type": "string",
//	        },
//	    },
//	}
//
//	err := ValidateAgainstSchema(`{"text": "Hello"}`, schema)
func ValidateAgainstSchema(response string, schema interface{}) error {
	var responseData interface{}
	if err := json.Unmarshal([]byte(response), &responseData); err != nil {
		return fmt.Errorf("failed to parse response JSON: %w", err)
	}

	var schemaMap map[string]interface{}
	switch s := schema.(type) {
	case string:
		if err := json.Unmarshal([]byte(s), &schemaMap); err != nil {
			return fmt.Errorf("failed to parse schema JSON string: %w", err)
		}
	case []byte:
		if err := json.Unmarshal(s, &schemaMap); err != nil {
			return fmt.Errorf("failed to parse schema JSON bytes: %w", err)
		}
	case map[string]interface{}:
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
func validateJSONAgainstSchema(data interface{}, schema map[string]interface{}) error {
	schemaType, ok := schema["type"].(string)
	if !ok {
		return fmt.Errorf("schema missing 'type' field")
	}

	switch schemaType {
	case "object":
		return validateObject(data, schema)
	case "array":
		return validateArray(data, schema)
	case "string", "number", "integer", "boolean":
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
func validateObject(data interface{}, schema map[string]interface{}) error {
	dataMap, ok := data.(map[string]interface{})
	if !ok {
		return fmt.Errorf("expected object, got %T", data)
	}

	properties, ok := schema["properties"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid 'properties' in schema")
	}

	for key, propSchema := range properties {
		propData, exists := dataMap[key]
		if !exists {
			if required, ok := schema["required"].([]interface{}); ok {
				for _, req := range required {
					if req.(string) == key {
						return fmt.Errorf("missing required field: %s", key)
					}
				}
			}
			continue
		}

		if err := validateJSONAgainstSchema(propData, propSchema.(map[string]interface{})); err != nil {
			return fmt.Errorf("invalid field '%s': %w", key, err)
		}
	}

	return nil
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
func validateArray(data interface{}, schema map[string]interface{}) error {
	dataSlice, ok := data.([]interface{})
	if !ok {
		return fmt.Errorf("expected array, got %T", data)
	}

	items, ok := schema["items"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid 'items' in schema")
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
func validatePrimitive(data interface{}, expectedType string) error {
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
