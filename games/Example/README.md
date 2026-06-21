# Example Ren'Py Game - Translation Demo

A minimal example Ren'Py visual novel used for testing the translation pipeline
end to end. It is the fixture for `tests/test_e2e_example.py`.

## Structure

```
Example/
├── README.md
└── game/
    ├── script.rpy                          # source Ren'Py script (English)
    └── tl/
        └── romanian/
            ├── Cell01_Academy.rpy                       # translation file (Ren'Py format)
            ├── Cell01_Academy.parsed.yaml               # extracted clean text (editable)
            ├── Cell01_Academy.tags.yaml                 # extracted tags + metadata
            ├── Cell01_Academy.translated.rpy            # merged output
            ├── Cell01_Academy.translated.corrections.txt# correction-pass log
            └── characters.yaml                          # character mappings
```

## What's Included

A short campus-introduction scene: a new student arrives at an academy and is
shown around by Sarah, who later introduces her friend Alex.

### Characters

- **narrator** - story narration
- **mc** - the player character, with variable name `[player_name]`
- **sarah** - a student who gives the tour
- **alex** - Sarah's friend

### Ren'Py features demonstrated

1. Basic dialogue and multi-line conversation
2. Variable substitution: `[player_name]`
3. Formatting tags: `{b}bold{/b}`, `{color=#ff69b4}colored{/color}`,
   `{size=18}sized{/size}`
4. Both dialogue blocks and string/menu-choice blocks

## Usage (Modular Pipeline)

From the repo root, after running `.\0-setup.ps1`:

```powershell
# 1. Configure this game (sets path, language, model)
.\1-config.ps1 -GamePath "games\Example" -Language ro -Model euroLLM9b

# 2. Extract clean text + tags
.\2-extract.ps1 -GameName Example -All
#    Creates: Cell01_Academy.parsed.yaml and Cell01_Academy.tags.yaml

# 3. Translate (or hand-edit the .parsed.yaml)
.\3-translate.ps1

# 4. (optional) Grammar/pattern correction
.\4-correct.ps1 "games\Example\game\tl\romanian"

# 5. Merge back to Ren'Py
.\5-merge.ps1 -GameName Example -All
#    Creates: Cell01_Academy.translated.rpy (tags restored, validated)
```

## Automated Test

```powershell
# Run the e2e example test directly
.\venv\Scripts\python.exe .\tests\test_e2e_example.py

# Or via the interactive runner
.\7-test.ps1
```

The test backs up the original translation file, runs the pipeline, verifies the
output, and restores the file to its original state.

## Notes

- All Ren'Py tags (`{b}`, `{color=...}`, etc.) are preserved exactly: stripped on
  extract, restored on merge.
- Square-bracket variables like `[player_name]` are left untranslated.
- Parsed YAML holds clean translatable text only; tag/structure metadata lives in
  the matching `.tags.yaml`.
