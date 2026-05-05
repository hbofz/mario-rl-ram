# SuperMarioBros-Nes-v0 Custom States

Place the World 5-2 savestate here:

```text
custom_integrations/SuperMarioBros-Nes-v0/Level5-2.state
```

Generate it with:

```bash
mario-capture-state --model models/<run>/<checkpoint>.zip --algo ppo
```

If the checkpoint cannot reach World 5-2 from `Level5-1`, create the same file with the Stable-Retro Integration UI. The training defaults validate that the state starts on displayed `5-2` before learning begins.

You can also play there manually:

```bash
mario-manual-capture-state
```

Controls are arrow keys, `Z` for jump, `X` for run/fire, Enter for Start, and Right Shift for Select. The script starts at `Level5-1` and auto-saves `Level5-2.state` after the game reports displayed World 5-2 for a short stretch.
