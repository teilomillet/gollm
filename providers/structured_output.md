```go
func schemaToVertex(fromObject map[string]any, parentObject map[string]any) (toObject map[string]any, err error) {
    toObject = make(map[string]any)
    
    fromAnyOf := getValueByPath(fromObject, []string{"anyOf"})
    if fromAnyOf != nil {
        setValueByPath(toObject, []string{"anyOf"}, fromAnyOf)
    }
    
    fromDefault := getValueByPath(fromObject, []string{"default"})
    if fromDefault != nil {
        setValueByPath(toObject, []string{"default"}, fromDefault)
    }
    
    fromDescription := getValueByPath(fromObject, []string{"description"})
    if fromDescription != nil {
        setValueByPath(toObject, []string{"description"}, fromDescription)
    }
    
    fromEnum := getValueByPath(fromObject, []string{"enum"})
    if fromEnum != nil {
        setValueByPath(toObject, []string{"enum"}, fromEnum)
    }
    
    fromExample := getValueByPath(fromObject, []string{"example"})
    if fromExample != nil {
        setValueByPath(toObject, []string{"example"}, fromExample)
    }
    
    fromFormat := getValueByPath(fromObject, []string{"format"})
    if fromFormat != nil {
        setValueByPath(toObject, []string{"format"}, fromFormat)
	}
    
    fromItems := getValueByPath(fromObject, []string{"items"})
    if fromItems != nil {
        setValueByPath(toObject, []string{"items"}, fromItems)
    }
    
    fromMaxItems := getValueByPath(fromObject, []string{"maxItems"})
    if fromMaxItems != nil {
        setValueByPath(toObject, []string{"maxItems"}, fromMaxItems)
	}
    
    fromMaxLength := getValueByPath(fromObject, []string{"maxLength"})
    if fromMaxLength != nil {
        setValueByPath(toObject, []string{"maxLength"}, fromMaxLength)
    }
    
    fromMaxProperties := getValueByPath(fromObject, []string{"maxProperties"})
    if fromMaxProperties != nil {
        setValueByPath(toObject, []string{"maxProperties"}, fromMaxProperties)
    }
    
    fromMaximum := getValueByPath(fromObject, []string{"maximum"})
    if fromMaximum != nil {
        setValueByPath(toObject, []string{"maximum"}, fromMaximum)
    }
    
    fromMinItems := getValueByPath(fromObject, []string{"minItems"})
    if fromMinItems != nil {
        setValueByPath(toObject, []string{"minItems"}, fromMinItems)
    }
    
    fromMinLength := getValueByPath(fromObject, []string{"minLength"})
    if fromMinLength != nil {
        setValueByPath(toObject, []string{"minLength"}, fromMinLength)
    }
    
    fromMinProperties := getValueByPath(fromObject, []string{"minProperties"})
    if fromMinProperties != nil {
        setValueByPath(toObject, []string{"minProperties"}, fromMinProperties)
    }
    
    fromMinimum := getValueByPath(fromObject, []string{"minimum"})
    if fromMinimum != nil {
        setValueByPath(toObject, []string{"minimum"}, fromMinimum)
    }
    
    fromNullable := getValueByPath(fromObject, []string{"nullable"})
    if fromNullable != nil {
        setValueByPath(toObject, []string{"nullable"}, fromNullable)
    }
    
    fromPattern := getValueByPath(fromObject, []string{"pattern"})
    if fromPattern != nil {
        setValueByPath(toObject, []string{"pattern"}, fromPattern)
    }
    
    fromProperties := getValueByPath(fromObject, []string{"properties"})
    if fromProperties != nil {
        setValueByPath(toObject, []string{"properties"}, fromProperties)
    }
    
    fromPropertyOrdering := getValueByPath(fromObject, []string{"propertyOrdering"})
    if fromPropertyOrdering != nil {
        setValueByPath(toObject, []string{"propertyOrdering"}, fromPropertyOrdering)
    }
    
    fromRequired := getValueByPath(fromObject, []string{"required"})
    if fromRequired != nil {
        setValueByPath(toObject, []string{"required"}, fromRequired)
    }
    
    fromTitle := getValueByPath(fromObject, []string{"title"})
    if fromTitle != nil {
        setValueByPath(toObject, []string{"title"}, fromTitle)
    }
    
    fromType := getValueByPath(fromObject, []string{"type"})
    if fromType != nil {
        setValueByPath(toObject, []string{"type"}, fromType)
    }
    
    return toObject, nil
}

// getValueByPath retrieves a value from a nested map or slice or struct based on a path of keys.
//
// Examples:
//
//	getValueByPath(map[string]any{"a": {"b": "v"}}, []string{"a", "b"})
//	  -> "v"
//	getValueByPath(map[string]any{"a": {"b": [{"c": "v1"}, {"c": "v2"}]}}, []string{"a", "b[]", "c"})
//	  -> []any{"v1", "v2"}
func getValueByPath(data any, keys []string) any {
if len(keys) == 1 && keys[0] == "_self" {
return data
}
if len(keys) == 0 {
return nil
}
var current any = data
for i, key := range keys {
if strings.HasSuffix(key, "[]") {
keyName := key[:len(key)-2]
switch v := current.(type) {
case map[string]any:
if sliceData, ok := v[keyName]; ok {
var result []any
switch concreteSliceData := sliceData.(type) {
case []map[string]any:
for _, d := range concreteSliceData {
result = append(result, getValueByPath(d, keys[i+1:]))
}
case []any:
for _, d := range concreteSliceData {
result = append(result, getValueByPath(d, keys[i+1:]))
}
default:
return nil
}
return result
} else {
return nil
}
default:
return nil
}
} else {
switch v := current.(type) {
case map[string]any:
current = v[key]
default:
return nil
}
}
}
return current
}

```