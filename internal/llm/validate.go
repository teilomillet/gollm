package llm

import (
	"encoding/json"
	"fmt"
	"reflect"
	"regexp"
	"strconv"
	"strings"

	"github.com/go-playground/validator/v10"
)

var validate *validator.Validate

func init() {
	validate = validator.New()
	validate.RegisterValidation("validGrade", validGrade)
}

// Validate checks if the given struct is valid according to its validation rules
func Validate(s interface{}) error {
	return validate.Struct(s)
}

// GenerateJSONSchema generates a JSON schema for the given struct
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
