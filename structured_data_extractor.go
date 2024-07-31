// File: structured_data_extractor.go

package gollm

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"regexp"
	"strconv"
	"strings"
)

// ExtractStructuredData extracts structured data from text based on a given struct type
func ExtractStructuredData[T any](ctx context.Context, l LLM, text string, opts ...PromptOption) (*T, error) {
	structType := reflect.TypeOf((*T)(nil)).Elem()
	schema, err := generateJSONSchemaFromStruct(structType, l)
	if err != nil {
		return nil, fmt.Errorf("failed to generate JSON schema: %w", err)
	}

	prompt := NewPrompt(
		fmt.Sprintf("Extract the following information from the given text:\n\n%s\n\nRespond with a JSON object matching this schema:\n%s", text, schema),
		append(opts,
			WithDirectives(
				"Extract all relevant information from the text",
				"Ensure the output matches the provided JSON schema exactly",
				"If a field cannot be confidently filled, leave it as null or an empty string/array as appropriate",
			),
			WithOutput("JSON object matching the provided schema"),
		)...,
	)

	response, err := l.Generate(ctx, prompt, WithJSONSchemaValidation())
	if err != nil {
		return nil, fmt.Errorf("failed to generate structured data: %w", err)
	}

	// Use the logger for debug output
	if logger, ok := l.(interface {
		Debug(msg string, keysAndValues ...interface{})
	}); ok {
		logger.Debug("Raw LLM Response", "response", response)
	}

	var result T
	if err := json.Unmarshal([]byte(response), &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Use the logger for debug output
	if logger, ok := l.(interface {
		Debug(msg string, keysAndValues ...interface{})
	}); ok {
		logger.Debug("Unmarshaled struct", "result", fmt.Sprintf("%+v", result))
	}

	// Validate the struct after unmarshaling
	if err := Validate(result); err != nil {
		return nil, fmt.Errorf("validation failed: %w", err)
	}

	return &result, nil
}

func generateJSONSchemaFromStruct(t reflect.Type, l LLM) (string, error) {
	schema := make(map[string]interface{})
	schema["type"] = "object"
	properties := make(map[string]interface{})
	required := []string{}

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
			return "", err
		}
		properties[jsonName] = fieldSchema

		// Check if the field is required
		if validateTag := field.Tag.Get("validate"); strings.Contains(validateTag, "required") {
			required = append(required, jsonName)
		}
	}

	schema["properties"] = properties
	if len(required) > 0 {
		schema["required"] = required
	}

	schemaJSON, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		return "", err
	}

	// Use the logger for debug output
	if logger, ok := l.(interface {
		Debug(msg string, keysAndValues ...interface{})
	}); ok {
		logger.Debug("Generated JSON Schema", "schema", string(schemaJSON))
	}

	return string(schemaJSON), nil
}

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
		properties := make(map[string]interface{})
		for i := 0; i < field.Type.NumField(); i++ {
			nestedField := field.Type.Field(i)
			jsonTag := nestedField.Tag.Get("json")
			if jsonTag == "-" {
				continue
			}
			jsonName := strings.Split(jsonTag, ",")[0]
			if jsonName == "" {
				jsonName = nestedField.Name
			}
			fieldSchema, err := getFieldSchema(nestedField)
			if err != nil {
				return nil, err
			}
			properties[jsonName] = fieldSchema
		}
		schema["properties"] = properties
	default:
		return nil, fmt.Errorf("unsupported type: %v", field.Type.Kind())
	}

	// Add validation rules to the schema
	addValidationToSchema(schema, field.Tag.Get("validate"))

	return schema, nil
}

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
				schema["minimum"] = num
			}

		case "max":
			if num, err := strconv.ParseFloat(value, 64); err == nil {
				schema["maximum"] = num
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
