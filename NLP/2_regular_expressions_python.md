# Regular Expressions in Python — A Step-by-Step Guide

> A practical, hands-on guide to mastering regex in Python using the `re` module.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Importing the `re` Module](#2-importing-the-re-module)
3. [Basic Patterns and Literals](#3-basic-patterns-and-literals)
4. [Metacharacters](#4-metacharacters)
5. [Character Classes](#5-character-classes)
6. [Quantifiers](#6-quantifiers)
7. [Anchors](#7-anchors)
8. [Groups and Capturing](#8-groups-and-capturing)
9. [Alternation](#9-alternation)
10. [Special Sequences](#10-special-sequences)
11. [Flags / Modifiers](#11-flags--modifiers)
12. [Core `re` Functions](#12-core-re-functions)
    - [re.match()](#remath)
    - [re.search()](#research)
    - [re.findall()](#refindall)
    - [re.finditer()](#refinditer)
    - [re.sub()](#resub)
    - [re.split()](#resplit)
    - [re.compile()](#recompile)
13. [Named Groups](#13-named-groups)
14. [Lookahead and Lookbehind](#14-lookahead-and-lookbehind)
15. [Non-Greedy Matching](#15-non-greedy-matching)
16. [Practical Examples](#16-practical-examples)
17. [Quick Reference Cheat Sheet](#17-quick-reference-cheat-sheet)

---

## 1. Introduction

A **regular expression** (regex) is a sequence of characters that defines a search pattern. Python's built-in `re` module provides full support for Perl-like regular expressions.

Regex is used for:
- Validating input (emails, phone numbers, URLs)
- Searching and extracting text
- Finding and replacing patterns
- Splitting strings by complex delimiters

---

## 2. Importing the `re` Module

```python
import re
```

No installation needed — `re` is part of Python's standard library.

---

## 3. Basic Patterns and Literals

The simplest regex is a plain string — it matches itself exactly.

```python
import re

text = "Hello, World!"

match = re.search("World", text)
print(match)        # <re.Match object; span=(7, 12), match='World'>
print(match.group()) # World
```

> **Note:** Regex is case-sensitive by default. `"world"` would NOT match `"World"`.

---

## 4. Metacharacters

Metacharacters have special meanings in regex. They must be **escaped with `\`** if you want to match them literally.

| Metacharacter | Meaning                          |
|---------------|----------------------------------|
| `.`           | Any character except newline     |
| `^`           | Start of string                  |
| `$`           | End of string                    |
| `*`           | 0 or more repetitions            |
| `+`           | 1 or more repetitions            |
| `?`           | 0 or 1 repetition                |
| `{}`          | Exact repetition count           |
| `[]`          | Character class                  |
| `\`           | Escape character                 |
| `\|`          | Alternation (OR)                 |
| `()`          | Grouping                         |

```python
import re

# '.' matches any single character
print(re.findall("h.t", "hat hit hot hut"))  # ['hat', 'hit', 'hot', 'hut']

# Escaping a literal dot
print(re.findall(r"3\.14", "pi is 3.14"))   # ['3.14']
```

> **Tip:** Always use raw strings (`r"..."`) for regex patterns to avoid Python's own escape interpretation.

---

## 5. Character Classes

Character classes let you match **one character from a defined set**.

```python
import re

text = "cat bat rat mat"

# Match any of: c, b, r, m followed by 'at'
print(re.findall(r"[cbrm]at", text))   # ['cat', 'bat', 'rat', 'mat']

# Range: lowercase letters
print(re.findall(r"[a-z]+", "Hello World 123"))  # ['ello', 'orld']

# Digits only
print(re.findall(r"[0-9]+", "Room 101 and 202"))  # ['101', '202']

# Negation: anything NOT a digit
print(re.findall(r"[^0-9]+", "abc123def456"))  # ['abc', 'def']
```

---

## 6. Quantifiers

Quantifiers define **how many times** a pattern should repeat.

| Quantifier | Meaning             | Example Pattern | Matches                   |
|------------|---------------------|-----------------|---------------------------|
| `*`        | 0 or more           | `ab*`           | `a`, `ab`, `abb`, `abbb`  |
| `+`        | 1 or more           | `ab+`           | `ab`, `abb`, `abbb`       |
| `?`        | 0 or 1              | `ab?`           | `a`, `ab`                 |
| `{n}`      | Exactly n           | `a{3}`          | `aaa`                     |
| `{n,}`     | n or more           | `a{2,}`         | `aa`, `aaa`, `aaaa`       |
| `{n,m}`    | Between n and m     | `a{2,4}`        | `aa`, `aaa`, `aaaa`       |

```python
import re

# One or more digits
print(re.findall(r"\d+", "I have 3 cats and 12 dogs"))  # ['3', '12']

# Exactly 3 digits
print(re.findall(r"\d{3}", "123 4567 89"))  # ['123', '456']

# Optional 's' (singular or plural)
print(re.findall(r"cats?", "I love cat and cats"))  # ['cat', 'cats']
```

---

## 7. Anchors

Anchors match a **position** in the string, not a character.

| Anchor | Meaning                        |
|--------|--------------------------------|
| `^`    | Start of string (or line)      |
| `$`    | End of string (or line)        |
| `\b`   | Word boundary                  |
| `\B`   | Non-word boundary              |

```python
import re

# ^ — must start with 'Hello'
print(re.search(r"^Hello", "Hello World"))   # Match
print(re.search(r"^Hello", "Say Hello"))     # None

# $ — must end with 'World'
print(re.search(r"World$", "Hello World"))   # Match
print(re.search(r"World$", "World Peace"))   # None

# \b — whole word match
print(re.findall(r"\bcat\b", "cat cats concatenate"))  # ['cat']
```

---

## 8. Groups and Capturing

Parentheses `()` create **capturing groups** that let you extract specific parts of a match.

```python
import re

# Basic group
match = re.search(r"(\d{4})-(\d{2})-(\d{2})", "Date: 2024-07-15")
if match:
    print(match.group(0))  # Full match: 2024-07-15
    print(match.group(1))  # Year:       2024
    print(match.group(2))  # Month:      07
    print(match.group(3))  # Day:        15

# groups() returns all captured groups as a tuple
print(match.groups())  # ('2024', '07', '15')
```

### Non-Capturing Groups

Use `(?:...)` when you need to group but don't need to capture:

```python
import re

# Groups 'ab' for repetition but doesn't capture
print(re.findall(r"(?:ab)+", "ab abab ababab"))  # ['ab', 'abab', 'ababab']
```

---

## 9. Alternation

The pipe `|` acts as an **OR** operator between patterns.

```python
import re

# Match 'cat' OR 'dog'
print(re.findall(r"cat|dog", "I have a cat and a dog"))  # ['cat', 'dog']

# Match multiple file extensions
files = "doc.pdf report.docx image.png data.csv"
print(re.findall(r"\w+\.(?:pdf|docx|png)", files))  # ['doc.pdf', 'report.docx', 'image.png']
```

---

## 10. Special Sequences

Python provides shorthand sequences for common character classes.

| Sequence | Equivalent       | Matches                          |
|----------|------------------|----------------------------------|
| `\d`     | `[0-9]`          | Any digit                        |
| `\D`     | `[^0-9]`         | Any non-digit                    |
| `\w`     | `[a-zA-Z0-9_]`   | Word character                   |
| `\W`     | `[^a-zA-Z0-9_]`  | Non-word character               |
| `\s`     | `[ \t\n\r\f\v]`  | Whitespace                       |
| `\S`     | `[^ \t\n\r\f\v]` | Non-whitespace                   |
| `\b`     | —                | Word boundary                    |
| `\B`     | —                | Non-word boundary                |

```python
import re

text = "User: alice123, Score: 98, Note: good work!"

# Extract all word characters
print(re.findall(r"\w+", text))
# ['User', 'alice123', 'Score', '98', 'Note', 'good', 'work']

# Extract all digits
print(re.findall(r"\d+", text))  # ['123', '98']

# Split on any whitespace
print(re.split(r"\s+", "Hello   World\tPython\nRegex"))
# ['Hello', 'World', 'Python', 'Regex']
```

---

## 11. Flags / Modifiers

Flags change the behaviour of the regex engine.

| Flag              | Short | Description                             |
|-------------------|-------|-----------------------------------------|
| `re.IGNORECASE`   | `re.I`| Case-insensitive matching               |
| `re.MULTILINE`    | `re.M`| `^` and `$` match start/end of each line|
| `re.DOTALL`       | `re.S`| `.` matches newline too                 |
| `re.VERBOSE`      | `re.X`| Allow comments and whitespace in pattern|

```python
import re

# Case-insensitive
print(re.findall(r"hello", "Hello HELLO hello", re.I))
# ['Hello', 'HELLO', 'hello']

# Multiline: ^ matches start of each line
text = "line one\nline two\nline three"
print(re.findall(r"^line", text, re.M))  # ['line', 'line', 'line']

# DOTALL: dot matches newlines too
print(re.search(r"start.*end", "start\nmiddle\nend", re.S).group())
# start\nmiddle\nend

# VERBOSE: write readable patterns with comments
pattern = re.compile(r"""
    (\d{3})   # area code
    [-.\s]?   # optional separator
    (\d{3})   # first 3 digits
    [-.\s]?   # optional separator
    (\d{4})   # last 4 digits
""", re.X)

print(pattern.findall("Call 800-555-1234 or 800.555.5678"))
# [('800', '555', '1234'), ('800', '555', '5678')]
```

---

## 12. Core `re` Functions

### `re.match()`

Matches **only at the beginning** of the string.

```python
import re

result = re.match(r"\d+", "123abc")
print(result.group())  # 123

result = re.match(r"\d+", "abc123")
print(result)  # None — doesn't start with digits
```

---

### `re.search()`

Scans the **entire string** and returns the first match.

```python
import re

result = re.search(r"\d+", "Room 42 on floor 7")
print(result.group())  # 42  (first match found)
print(result.span())   # (5, 7)
```

---

### `re.findall()`

Returns a **list of all non-overlapping matches**.

```python
import re

emails = "Contact alice@example.com or bob@test.org for info"
pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

print(re.findall(pattern, emails))
# ['alice@example.com', 'bob@test.org']
```

---

### `re.finditer()`

Like `findall()`, but returns an **iterator of match objects** (useful for large texts or when you need span info).

```python
import re

text = "apple 5, banana 12, cherry 3"

for match in re.finditer(r"\d+", text):
    print(f"Found '{match.group()}' at position {match.start()}-{match.end()}")

# Found '5' at position 6-7
# Found '12' at position 17-19
# Found '3' at position 28-29
```

---

### `re.sub()`

**Replaces** all matches with a replacement string or function.

```python
import re

# Simple replacement
result = re.sub(r"\bfoo\b", "bar", "foo is not foobar, just foo")
print(result)  # bar is not foobar, just bar

# Using a function as replacement
def double_number(match):
    return str(int(match.group()) * 2)

result = re.sub(r"\d+", double_number, "I have 3 cats and 5 dogs")
print(result)  # I have 6 cats and 10 dogs

# Limit replacements with count parameter
result = re.sub(r"a", "X", "banana", count=2)
print(result)  # bXnXna
```

---

### `re.split()`

**Splits** a string by the pattern.

```python
import re

# Split by one or more spaces
print(re.split(r"\s+", "Hello   World   Python"))
# ['Hello', 'World', 'Python']

# Split by comma, semicolon, or pipe
print(re.split(r"[,;|]", "one,two;three|four"))
# ['one', 'two', 'three', 'four']

# Keep the delimiter in output using a capturing group
print(re.split(r"(\s+)", "Hello World Python"))
# ['Hello', ' ', 'World', ' ', 'Python']
```

---

### `re.compile()`

**Pre-compiles** a pattern into a regex object for reuse — improves performance when using the same pattern multiple times.

```python
import re

# Compile once, use many times
phone_pattern = re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")

contacts = [
    "Alice: 800-555-1234",
    "Bob: 212.555.9876",
    "Charlie: no phone listed",
]

for contact in contacts:
    match = phone_pattern.search(contact)
    if match:
        print(f"Phone found: {match.group()}")

# Phone found: 800-555-1234
# Phone found: 212.555.9876
```

---

## 13. Named Groups

Named groups (`(?P<name>...)`) make patterns more readable and allow access by name.

```python
import re

pattern = r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
match = re.search(pattern, "Event date: 2024-12-25")

if match:
    print(match.group("year"))   # 2024
    print(match.group("month"))  # 12
    print(match.group("day"))    # 25
    print(match.groupdict())     # {'year': '2024', 'month': '12', 'day': '25'}
```

Named groups can also be referenced in `re.sub()`:

```python
import re

# Reformat date from YYYY-MM-DD to DD/MM/YYYY
result = re.sub(
    r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})",
    r"\g<day>/\g<month>/\g<year>",
    "Date: 2024-07-15"
)
print(result)  # Date: 15/07/2024
```

---

## 14. Lookahead and Lookbehind

These are **zero-width assertions** — they match a position based on what's around it, without consuming characters.

| Syntax      | Type               | Description                              |
|-------------|--------------------|------------------------------------------|
| `(?=...)`   | Positive lookahead | Match if followed by `...`               |
| `(?!...)`   | Negative lookahead | Match if NOT followed by `...`           |
| `(?<=...)`  | Positive lookbehind| Match if preceded by `...`               |
| `(?<!...)`  | Negative lookbehind| Match if NOT preceded by `...`           |

```python
import re

# Positive lookahead: digits followed by 'px'
print(re.findall(r"\d+(?=px)", "font: 14px, margin: 20px, opacity: 0.5"))
# ['14', '20']

# Negative lookahead: digits NOT followed by 'px'
print(re.findall(r"\d+(?!px)", "font: 14px, count: 42"))
# ['1', '42']  ← note: '1' from '14' since '4px' matches, '1' doesn't

# Positive lookbehind: word after '$'
print(re.findall(r"(?<=\$)\d+", "Total: $150, Tax: $12"))
# ['150', '12']

# Negative lookbehind: word NOT after '#'
print(re.findall(r"(?<!#)\b\w+", "normal #tag word"))
# Finds words not immediately preceded by '#'
```

---

## 15. Non-Greedy Matching

By default, quantifiers are **greedy** — they match as much as possible. Adding `?` makes them **non-greedy** (lazy), matching as little as possible.

```python
import re

html = "<b>Bold</b> and <i>italic</i>"

# Greedy: matches from first '<' to last '>'
print(re.findall(r"<.+>", html))
# ['<b>Bold</b> and <i>italic</i>']

# Non-greedy: matches each tag individually
print(re.findall(r"<.+?>", html))
# ['<b>', '</b>', '<i>', '</i>']
```

| Greedy | Non-Greedy | Meaning             |
|--------|------------|---------------------|
| `*`    | `*?`       | 0 or more (lazy)    |
| `+`    | `+?`       | 1 or more (lazy)    |
| `?`    | `??`       | 0 or 1 (lazy)       |
| `{n,m}`| `{n,m}?`  | n to m (lazy)       |

---

## 16. Practical Examples

### Validate an Email Address

```python
import re

def is_valid_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))

print(is_valid_email("user@example.com"))   # True
print(is_valid_email("bad-email@.com"))     # False
print(is_valid_email("no_at_sign.com"))     # False
```

---

### Extract URLs from Text

```python
import re

text = "Visit https://www.python.org or http://docs.python.org/3/ for docs."

url_pattern = r"https?://[^\s]+"
urls = re.findall(url_pattern, text)

for url in urls:
    print(url)
# https://www.python.org
# http://docs.python.org/3/
```

---

### Validate a Password

```python
import re

def validate_password(password):
    """
    Must be at least 8 chars, contain:
    - One uppercase letter
    - One lowercase letter
    - One digit
    - One special character
    """
    checks = {
        "Min 8 characters":      r".{8,}",
        "Uppercase letter":      r"[A-Z]",
        "Lowercase letter":      r"[a-z]",
        "Digit":                 r"\d",
        "Special character":     r"[!@#$%^&*(),.?\":{}|<>]",
    }

    for rule, pattern in checks.items():
        if not re.search(pattern, password):
            print(f"  ✗ Missing: {rule}")
            return False

    print("  ✓ Password is valid!")
    return True

validate_password("weakpass")       # Fails multiple checks
validate_password("Str0ng!Pass")    # Passes all checks
```

---

### Parse Log Files

```python
import re

log_lines = [
    "2024-07-15 08:23:41 ERROR Failed to connect to database",
    "2024-07-15 08:24:10 INFO  Server started on port 8080",
    "2024-07-15 08:25:05 WARN  High memory usage detected",
]

log_pattern = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})\s"
    r"(?P<time>\d{2}:\d{2}:\d{2})\s"
    r"(?P<level>\w+)\s+"
    r"(?P<message>.+)"
)

for line in log_lines:
    match = log_pattern.match(line)
    if match:
        info = match.groupdict()
        print(f"[{info['level']}] {info['date']} {info['time']} — {info['message']}")

# [ERROR] 2024-07-15 08:23:41 — Failed to connect to database
# [INFO] 2024-07-15 08:24:10 — Server started on port 8080
# [WARN] 2024-07-15 08:25:05 — High memory usage detected
```

---

## 17. Quick Reference Cheat Sheet

```
CHARACTERS
  .         Any character except newline
  \d        Digit [0-9]
  \D        Non-digit
  \w        Word character [a-zA-Z0-9_]
  \W        Non-word character
  \s        Whitespace
  \S        Non-whitespace
  \b        Word boundary
  \B        Non-word boundary

CHARACTER CLASSES
  [abc]     Any of a, b, or c
  [^abc]    Anything except a, b, or c
  [a-z]     Any lowercase letter
  [A-Z]     Any uppercase letter
  [0-9]     Any digit
  [a-zA-Z]  Any letter

QUANTIFIERS (Greedy)
  *         0 or more
  +         1 or more
  ?         0 or 1
  {n}       Exactly n
  {n,}      n or more
  {n,m}     Between n and m

QUANTIFIERS (Non-Greedy / Lazy)
  *?  +?  ??  {n,m}?

ANCHORS
  ^         Start of string/line
  $         End of string/line
  \b        Word boundary
  \A        Start of string
  \Z        End of string

GROUPS
  (...)     Capturing group
  (?:...)   Non-capturing group
  (?P<name>...)  Named group
  (?P=name)      Backreference to named group

LOOKAROUND
  (?=...)   Positive lookahead
  (?!...)   Negative lookahead
  (?<=...)  Positive lookbehind
  (?<!...)  Negative lookbehind

FLAGS
  re.I / re.IGNORECASE   Case-insensitive
  re.M / re.MULTILINE    ^ and $ match each line
  re.S / re.DOTALL       . matches newline
  re.X / re.VERBOSE      Allow whitespace and comments

FUNCTIONS
  re.match(pat, str)      Match at beginning only
  re.search(pat, str)     Search entire string
  re.findall(pat, str)    Return list of all matches
  re.finditer(pat, str)   Return iterator of match objects
  re.sub(pat, repl, str)  Replace matches
  re.split(pat, str)      Split string by pattern
  re.compile(pat)         Pre-compile pattern
```

---

*Happy pattern matching! 🐍*
